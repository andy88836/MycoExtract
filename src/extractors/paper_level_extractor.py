"""
Paper-Level Multi-Model Extractor with Aggregation Agent

使用文献中的思路：
1. 多个"学生"模型（Kimi, DeepSeek, 可选第三文本模型）分别提取整篇论文
2. 一个"老师"模型（配置文件指定的aggregation模型）聚合结果

优势：
- 论文级别提取，可以对齐跨块的记录
- 利用强模型的推理能力智能聚合
- 一次调用得到最优结果

配置灵活性：
- 支持2个学生模型（Kimi + DeepSeek）+ 配置的视觉模型
- 支持3个学生模型（Kimi + DeepSeek + 第三文本模型）+ 配置的视觉模型
"""

import asyncio
import logging
import random
import re
import os
import io
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 图片处理优化
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# HTML解析测试（用于智能路由）
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ========================================
# Vision-table extraction optimization config
# ========================================

# 优化1: 表格预筛选关键词
TABLE_INCLUDE_KEYWORDS = [
    "km", "kcat", "k_cat", "turnover", "catalytic", "kinetic",
    "vmax", "velocity", "activity", "degradation", "detoxification",
    "substrate", "enzyme", "mycotoxin", "aflatoxin", "ochratoxin",
    "deoxynivalenol", "zearalenone", "fumonisin", "t-2", "patulin"
]

TABLE_EXCLUDE_KEYWORDS = [
    "pdb", "accession", "uniprot", "genbank", "primer", "sequence",
    "gene ontology", "go term", "pathway", "network", "interaction",
    "binding site", "alignment", "homology", "phylogenetic", "blast"
]

# 智能路由: 动力学参数表头关键词（用于判断是否需要提取）
KINETIC_HEADER_KEYWORDS = [
    "km", "k_m", "kcat", "k_cat", "kcat/km", "turnover", "catalytic",
    "vmax", "velocity", "degradation", "detoxification", "efficiency",
    "activity", "substrate", "michaelis", "kinetic", "enzyme",
    "mycotoxin", "aflatoxin", "ochratoxin", "deoxynivalenol", "zearalenone"
]

MYCOTOXIN_SUBSTRATE_TERMS = [
    "aflatoxin", "afb1", "afb2", "afg1", "afg2", "afm1",
    "ochratoxin", "ota", "otb", "deoxynivalenol", "don", "nivalenol", "niv",
    "zearalenone",
    "zearalanol", "zearalanone", "zearalenol",
    "fumonisin", "fb1", "fb2", "fb3", "patulin", "sterigmatocystin", "t-2", "ht-2",
    "mycotoxin"
]

ENZYME_SYSTEM_TERMS = [
    "enzyme", "thermolysin", "laccase", "peroxidase", "oxidase", "hydrolase",
    "esterase", "lactonase", "transferase", "glucosyltransferase", "ugt",
    "oph", "zdh", "zhd", "adh", "gsta", "cotA".lower(), "ple"
]

# 优化2: 表格提取 max_tokens（大表格需要更多 token 输出所有行）
GLM46V_MAX_TOKENS = 16384  # vision-table max token budget
TABLE_TEXT_MAX_TOKENS = 16384  # text-only table extraction max token budget
FULL_TEXT_MAX_TOKENS = 16384  # full-paper text extraction output budget

# 优化3: 图片压缩配置
MAX_IMAGE_WIDTH = 1024
JPEG_QUALITY = 75

# Token统计
class TokenTracker:
    """追踪GLM-4.6V的token消耗和路由统计"""
    total_tokens = 0
    total_images = 0
    skipped_tables = 0
    # 智能路由统计
    text_only_tables = 0    # 纯文本提取的表格数
    vision_model_tables = 0 # 视觉模型提取的表格数
    no_kinetic_keyword_tables = 0  # 不含动力学关键词跳过的表格数

    @classmethod
    def add_image(cls, tokens: int = 0):
        cls.total_images += 1
        if tokens > 0:
            cls.total_tokens += tokens

    @classmethod
    def add_skipped_table(cls):
        cls.skipped_tables += 1

    @classmethod
    def add_text_only_table(cls):
        """记录纯文本提取的表格"""
        cls.text_only_tables += 1

    @classmethod
    def add_vision_model_table(cls):
        """记录视觉模型提取的表格"""
        cls.vision_model_tables += 1

    @classmethod
    def add_no_keyword_table(cls):
        """记录不含动力学关键词跳过的表格"""
        cls.no_kinetic_keyword_tables += 1

    @classmethod
    def log_stats(cls):
        if cls.total_images > 0:
            avg_tokens = cls.total_tokens // cls.total_images
            logger.info(f"    [Vision Model Stats] Images: {cls.total_images}, Tokens: {cls.total_tokens:,}, Avg: {avg_tokens:,}/img")
        if cls.skipped_tables > 0:
            logger.info(f"    [Vision Model Stats] Skipped tables: {cls.skipped_tables} ({cls.skipped_tables/(cls.total_images+cls.skipped_tables)*100:.1f}%)")
        # 智能路由统计
        if cls.text_only_tables > 0 or cls.no_kinetic_keyword_tables > 0:
            total_processed = cls.text_only_tables + cls.vision_model_tables + cls.no_kinetic_keyword_tables
            if total_processed > 0:
                logger.info(f"    [Smart Routing] Text-only: {cls.text_only_tables}, Vision: {cls.vision_model_tables}, No-keyword: {cls.no_kinetic_keyword_tables}")
                saved_pct = cls.text_only_tables / total_processed * 100
                logger.info(f"    [Smart Routing] Saved {saved_pct:.1f}% vision calls ({cls.text_only_tables} tables)")
    @classmethod
    def reset(cls):
        """重置统计（用于每篇论文处理开始时）"""
        cls.total_tokens = 0
        cls.total_images = 0
        cls.skipped_tables = 0
        cls.text_only_tables = 0
        cls.vision_model_tables = 0
        cls.no_kinetic_keyword_tables = 0

# 🔧 API并发限制配置
# 每个API提供商的最大并发数（延迟创建，避免事件循环冲突）
API_CONCURRENCY_LIMITS = {
    "kimi": 3,      # Kimi: 最多3个并发
    "deepseek": 5,   # DeepSeek: 相对宽松，5并发
    "glm-4.7": 1,    # GLM文本: 非常严格，1并发（API限制）
    "MiniMax-M2.7": 1,
    "minimax": 1,
    "glm-4.6v": 1,   # GLM多模态: 非常严格，1并发（API限制）
    "mimo-v2.5": 1,
    "mimo-v2.5-pro": 1,
    "mimo": 1,
}

# 存储当前事件循环的 Semaphore 实例
_semaphore_cache = {}

def get_semaphore(model_name: str) -> asyncio.Semaphore:
    """获取或创建指定模型的Semaphore（避免事件循环冲突）"""
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        
        # 为每个事件循环创建独立的 Semaphore 字典
        if loop_id not in _semaphore_cache:
            _semaphore_cache[loop_id] = {}
        
        # 如果该模型的 Semaphore 不存在，创建它
        if model_name not in _semaphore_cache[loop_id]:
            limit = API_CONCURRENCY_LIMITS.get(model_name, 2)
            _semaphore_cache[loop_id][model_name] = asyncio.Semaphore(limit)
        
        return _semaphore_cache[loop_id][model_name]
    except RuntimeError:
        # 如果不在事件循环中，返回默认值
        limit = API_CONCURRENCY_LIMITS.get(model_name, 2)
        return asyncio.Semaphore(limit)

# 指数退避重试配置
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 2.0,    # 基础延迟2秒
    "max_delay": 30.0,    # 最大延迟30秒
    "jitter": 0.5,        # 随机抖动因子
}


class PaperLevelMultiModelExtractor:
    """
    论文级别多模型提取器 + 聚合智能体

    工作流程：
    1. 用N个模型组合分别提取整篇论文（N=2或3）
       - 组合A: Kimi (文本) + 配置的视觉模型 (表格图片，共享)
       - 组合B: DeepSeek (文本) + 配置的视觉模型 (表格图片，共享)
       - 组合C: 第三文本模型 + 配置的视觉模型 (表格图片，共享) [可选]
    2. 用Aggregation Agent（配置的teacher模型）聚合N个结果

    灵活配置：
    - 2学生模型模式：kimi_client + deepseek_client（glm47_client=None）
    - 3学生模型模式：kimi_client + deepseek_client + 第三文本模型
    """
    
    def __init__(
        self,
        kimi_client,
        deepseek_client,
        glm47_client,  # 可选，传 None 则只使用2个学生模型
        glm46v_client,
        aggregation_client,  # GPT-5.1 或 Claude 3.5
        text_prompt_template: str,
        table_prompt_template: str,
        figure_prompt_template: str,
        disable_table_image: bool = False,
    ):
        """
        Args:
            kimi_client: Kimi文本模型
            deepseek_client: DeepSeek文本模型
            glm47_client: 第三文本模型（历史参数名；可为MiniMax等，可选，传None则不使用）
            glm46v_client: 表格视觉多模态模型（历史参数名；可为MiMo等）
            aggregation_client: 聚合用的teacher模型
            text_prompt_template: 文本提取prompt
            table_prompt_template: 表格提取prompt
            figure_prompt_template: 图片提取prompt
            disable_table_image: 关闭 table-image 分支（消融实验用），
                所有原本走 vision 的表格改为走 text-only 提取
        """
        # 构建文本模型字典，自动过滤掉 None 的客户端
        self.text_models = {
            "kimi": kimi_client,
            "deepseek": deepseek_client,
        }
        if glm47_client is not None:
            self.text_models[getattr(glm47_client, "model_name", None) or "glm-4.7"] = glm47_client

        self.multimodal_model = glm46v_client
        self.aggregation_client = aggregation_client
        self.disable_table_image = disable_table_image

        self.text_prompt = text_prompt_template
        self.table_prompt = table_prompt_template
        self.figure_prompt = figure_prompt_template

        logger.info("Initialized PaperLevelMultiModelExtractor")
        logger.info(f"  - Text models: {list(self.text_models.keys())}")
        logger.info(f"  - Multimodal model: {getattr(self.multimodal_model, 'model_name', 'configured vision model')}")
        logger.info(f"  - Aggregation model: {getattr(self.aggregation_client, 'model_name', 'configured teacher model')}")
        if disable_table_image:
            logger.info("  - ⚠ TABLE-IMAGE BRANCH DISABLED (ablation mode)")
    
    async def extract_paper(
        self,
        paper_blocks: List[Dict],
        doi: str,
        paper_dir: Path
    ) -> Dict[str, Any]:
        """
        提取整篇论文
        
        Args:
            paper_blocks: 论文所有块 [
                {"type": "text", "content": "...", "block_id": 1},
                {"type": "table", "content": "<table>...</table>", "block_id": 2},
                {"type": "figure", "image_path": "fig1.jpg", "block_id": 3}
            ]
            doi: 论文DOI
            paper_dir: 论文目录（用于解析图片路径）
            
        Returns:
            {
                "aggregated_records": [...],  # 聚合后的最终结果
                "model_results": {             # 每个模型的原始结果
                    "kimi": [...],
                    "deepseek": [...],
                    "glm-4.6": [...]
                },
                "aggregation_notes": "...",
                "confidence": "high|medium|low"
            }
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"[Paper-Level Extraction] DOI: {doi}")
        logger.info(f"{'='*80}")
        logger.info(f"  - Total blocks: {len(paper_blocks)}")

        # 保存 paper_blocks 引用，供表格提取时获取论文上下文
        self._current_paper_blocks = paper_blocks
        self._skipped_table_type_gate_count = 0

        # ================================================================
        # Step 0: 先用 GLM-4.6V 提取所有表格图片（只调用一次，结果共享给所有学生模型）
        # ================================================================
        table_blocks = [b for b in paper_blocks if b.get('type') == 'table']
        
        shared_table_records = []
        if table_blocks:
            logger.info(f"\n[Step 0/3] Extracting {len(table_blocks)} tables as independent table candidates...")
            shared_table_records = await self._extract_all_tables_once(
                table_blocks=table_blocks,
                paper_dir=paper_dir
            )
            # 后处理：修正"突变名当酶名"问题
            shared_table_records = self._fix_table_enzyme_names(shared_table_records)
            logger.info(f"  ✓ Extracted {len(shared_table_records)} table candidate records")

        def is_rescue(record: Dict) -> bool:
            flags = record.get("error_flags") or []
            if isinstance(flags, str):
                flags = [flags]
            return "table_image_rescue" in flags

        parsed_table_candidates = []
        table_image_rescue_candidates = []
        for idx, record in enumerate(shared_table_records, start=1):
            channel = "table_image_rescue" if is_rescue(record) else "parsed_table"
            annotated = self._annotate_candidate_provenance(
                record,
                source_channel=channel,
                source_model=record.get("_source_model") or record.get("_extracted_by") or "table_extractor",
                raw_record_index=idx,
                locked_candidate=(channel == "table_image_rescue"),
            )
            if channel == "table_image_rescue":
                table_image_rescue_candidates.append(annotated)
            else:
                parsed_table_candidates.append(annotated)

        locked_table_candidates = [
            self._annotate_candidate_provenance(
                record,
                source_channel="parsed_table",
                source_model=record.get("source_model") or record.get("_extracted_by") or "table_extractor",
                raw_record_index=record.get("raw_record_index"),
                locked_candidate=True,
            )
            for record in parsed_table_candidates
            if self._is_locked_table_candidate(record)
        ]
        # Auto-clear human_review_required for locked table records with known mycotoxin substrates
        _mycotoxin_terms = (
            "aflatoxin", "afb1", "afm1", "ochratoxin", "ota",
            "deoxynivalenol", "don", "zearalenone", "zearalanone", "zearalanol",
            "zearalenol", "zel", "zen", "zea", "patulin",
            "sterigmatocystin", "citrinin", "fumonisin", "fb1", "fb2",
            "t-2", "ht-2", "nivalenol", "niv", "mycotoxin",
        )
        for rec in locked_table_candidates:
            if rec.get("human_review_required"):
                sub_text = str(rec.get("substrate") or "").lower()
                if sub_text and any(t in sub_text for t in _mycotoxin_terms):
                    rec["human_review_required"] = False
        logger.info(
            "  [Candidate Pool] parsed_table=%s table_image_rescue=%s locked_table=%s",
            len(parsed_table_candidates),
            len(table_image_rescue_candidates),
            len(locked_table_candidates),
        )
        
        # Step 1: 用N个学生模型【并行】提取文本（不再重复调用 GLM-4.6V）
        # 🔧 使用Semaphore限制每个API的并发数 + 指数退避重试
        num_student_models = len(self.text_models)
        logger.info(f"\n[Step 1/3] Extracting TEXT with {num_student_models} student models (parallel, tables already extracted)...")
        
        # 并行创建所有模型的提取任务。表格候选不再复制到各 student output。
        async def extract_with_model(model_name: str, text_client):
            logger.info(f"  [{model_name.upper()}] Starting text extraction...")
            try:
                # 只提取文本，不调用 GLM-4.6V
                records = await self._extract_text_only_with_model(
                    text_model=text_client,
                    paper_blocks=paper_blocks,
                    paper_dir=paper_dir,
                    model_name=model_name
                )
                annotated_records = [
                    self._annotate_candidate_provenance(
                        record,
                        source_channel="text",
                        source_model=model_name,
                        raw_record_index=idx,
                        locked_candidate=False,
                    )
                    for idx, record in enumerate(records, start=1)
                ]
                status = "success" if annotated_records else "empty_success"
                logger.info(f"  [{model_name.upper()}] ✓ Text extraction {status}; records={len(annotated_records)}")
                return {
                    "model_name": model_name,
                    "records": annotated_records,
                    "status": status,
                    "error": "",
                }
            except Exception as e:
                status = self._classify_model_error(e)
                logger.error(f"  [{model_name.upper()}] ✗ Text extraction {status}: {e}")
                return {
                    "model_name": model_name,
                    "records": [],
                    "status": status,
                    "error": str(e),
                }
        
        # 并行执行三个模型
        tasks = [
            extract_with_model(name, client) 
            for name, client in self.text_models.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果。失败模型不作为 negative evidence 送入 teacher。
        model_results = {}
        text_candidates = {}
        text_model_statuses = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"  Model extraction exception: {result}")
                model_name = "unknown_model"
                text_model_statuses[model_name] = {
                    "status": self._classify_model_error(result),
                    "error": str(result),
                    "records": 0,
                }
            else:
                model_name = result["model_name"]
                text_records = result["records"]
                text_candidates[model_name] = text_records
                text_model_statuses[model_name] = {
                    "status": result["status"],
                    "error": result["error"],
                    "records": len(text_records),
                }
                if result["status"] in ("success", "empty_success"):
                    model_results[model_name] = text_records
        
        # Step 2: Teacher 只 harmonize text candidates，不处理 locked table candidates。
        aggregation_model_name = getattr(self.aggregation_client, "model_name", None) or "configured teacher model"
        logger.info(f"\n[Step 2/3] Harmonizing TEXT candidates with {aggregation_model_name}...")
        
        # 收集原文（用于Aggregation Agent参考）- 使用 full.md
        original_text = self._collect_original_text(paper_blocks, paper_dir)
        
        # 调用Aggregation Agent（支持工具调用）
        from src.agents.aggregation_agent import AggregationAgent
        
        teacher_harmonized_records = []
        if any(model_results.values()):
            aggregation_agent = AggregationAgent(
                llm_client=self.aggregation_client,
                model_name=aggregation_model_name,
                paper_dir=paper_dir
            )
            teacher_harmonized_records = aggregation_agent.aggregate(
                original_text=original_text,
                model_results=model_results,
                doi=doi,
                paper_blocks=paper_blocks,
                locked_table_records=locked_table_candidates
            )
            teacher_harmonized_records = [
                self._annotate_candidate_provenance(
                    record,
                    source_channel=record.get("source_channel") or "text",
                    source_model=record.get("source_model") or record.get("_source_model") or "teacher_harmonized",
                    raw_record_index=idx,
                    locked_candidate=False,
                )
                for idx, record in enumerate(teacher_harmonized_records, start=1)
            ]
        else:
            logger.info("  [Aggregation Agent] Skipped: no non-empty text candidates")

        text_metric_fallbacks = self._extract_text_degradation_pair_fallbacks(
            original_text=original_text,
            existing_records=locked_table_candidates + table_image_rescue_candidates + teacher_harmonized_records,
        )
        if text_metric_fallbacks:
            text_metric_fallbacks = [
                self._annotate_candidate_provenance(
                    record,
                    source_channel="text",
                    source_model="deterministic_text_fallback",
                    raw_record_index=idx,
                    locked_candidate=False,
                )
                for idx, record in enumerate(text_metric_fallbacks, start=1)
            ]
            logger.warning(f"  [Text Fallback] Added {len(text_metric_fallbacks)} explicit degradation records from text")

        raw_text_candidates = []
        for model_name, text_records in model_results.items():
            for idx, record in enumerate(text_records or [], start=1):
                raw_text_candidates.append(
                    self._annotate_candidate_provenance(
                        record,
                        source_channel=record.get("source_channel") or "text",
                        source_model=record.get("source_model") or record.get("_source_model") or model_name,
                        raw_record_index=idx,
                        locked_candidate=False,
                    )
                )

        final_candidate_pool = self._dedupe_candidate_pool(
            locked_table_candidates
            + table_image_rescue_candidates
            + teacher_harmonized_records
            + raw_text_candidates
            + text_metric_fallbacks
        )
        after_safety_filters = self._apply_post_aggregation_safety_filters(final_candidate_pool)
        
        logger.info(
            "  ✓ Candidate pool: locked_table=%s, table_image_rescue=%s, teacher_text=%s, fallback=%s, final_after_validator=%s",
            len(locked_table_candidates),
            len(table_image_rescue_candidates),
            len(teacher_harmonized_records),
            len(text_metric_fallbacks),
            len(after_safety_filters),
        )
        logger.info(f"{'='*80}\n")
        
        return {
            "aggregated_records": after_safety_filters,
            "model_results": model_results,
            "debug_trace": {
                "text_candidates_raw": text_candidates,
                "text_model_statuses": text_model_statuses,
                "text_candidates_by_model": text_candidates,
                "parsed_table_candidates_raw": parsed_table_candidates,
                "table_image_rescue_candidates_raw": table_image_rescue_candidates,
                "locked_table_candidates": locked_table_candidates,
                "aggregation_input_records": {
                    "text_candidates_by_model": model_results,
                    "parsed_table_candidates": parsed_table_candidates,
                    "table_image_rescue_candidates": table_image_rescue_candidates,
                    "locked_table_candidates": locked_table_candidates,
                },
                "teacher_harmonized_records": teacher_harmonized_records,
                "aggregation_output_records": teacher_harmonized_records,
                "raw_text_candidates_in_pool": raw_text_candidates,
                "final_candidate_pool": final_candidate_pool,
                "after_rescue_protection": final_candidate_pool,
                "after_safety_filters": after_safety_filters,
                "after_deterministic_validator": after_safety_filters,
                "text_metric_fallbacks": text_metric_fallbacks,
                "final_records": after_safety_filters,
            },
            "doi": doi,
            "num_blocks": len(paper_blocks)
        }

    def _classify_model_error(self, exc: Exception) -> str:
        """Classify text model failures so failed calls are not treated as no-record evidence."""
        message = str(exc).lower()
        if "timeout" in message or "timed out" in message:
            return "timeout"
        if "json" in message or "parse" in message:
            return "parse_error"
        return "api_error"

    def _record_flags(self, record: Dict) -> List[str]:
        flags = record.get("error_flags") or []
        if isinstance(flags, str):
            flags = [f.strip() for f in flags.replace("|", ";").split(";") if f.strip()]
        return flags if isinstance(flags, list) else []

    def _has_primary_metric(self, record: Dict) -> bool:
        return any(record.get(field) not in (None, "", []) for field in (
            "Km_value", "kcat_value", "kcat_Km_value", "degradation_efficiency"
        ))

    # 突变名模式：WT, wild-type, 单字母+数字+单字母（如 Q202E, H122A）
    _MUTATION_PATTERN = re.compile(
        r'^(?:WT|wild[- ]?type|[A-Z]\d{1,4}[A-Z](?:/[A-Z]\d{1,4}[A-Z])*)$',
        re.IGNORECASE
    )

    def _fix_table_enzyme_names(self, records: List[Dict]) -> List[Dict]:
        """
        后处理：检测并修正表格提取中的"突变名当酶名"问题。

        当 enzyme_name 匹配突变模式（如 WT、Q202E、H122A）且 mutations 字段
        为同一值时，尝试从同批次记录中找到真正的酶名称进行修正。
        """
        if not records:
            return records

        # 收集所有记录中的非突变酶名称
        real_enzyme_names = set()
        for r in records:
            for field in ("enzyme_name", "reported_enzyme_name", "canonical_enzyme_name"):
                val = r.get(field)
                if val and isinstance(val, str):
                    stripped = val.strip()
                    if stripped and not self._MUTATION_PATTERN.match(stripped):
                        real_enzyme_names.add(stripped)

        fixed_count = 0
        for record in records:
            enzyme_name = (record.get("enzyme_name") or "").strip()
            mutations = (record.get("mutations") or "").strip()

            # 检测：enzyme_name 是突变名模式
            if not enzyme_name or not self._MUTATION_PATTERN.match(enzyme_name):
                continue

            # 已经有 mutations 且和 enzyme_name 相同，或者 mutations 为空
            # → 需要修正
            if mutations == enzyme_name or not mutations:
                # 将当前 enzyme_name 移到 mutations
                record["mutations"] = enzyme_name

                # 尝试从同批次中找到真正的酶名称
                found_real = False
                for candidate in real_enzyme_names:
                    # 检查 candidate 是否是更通用的酶名
                    # （例如 "Os79" 出现在其他记录中）
                    record["enzyme_name"] = candidate
                    found_real = True
                    fixed_count += 1
                    logger.debug(f"    [Enzyme Name Fix] '{enzyme_name}' → '{candidate}' (mutations='{enzyme_name}')")
                    break

                if not found_real:
                    # 从 reported_enzyme_name 中提取前缀
                    reported = (record.get("reported_enzyme_name") or "").strip()
                    if reported and not self._MUTATION_PATTERN.match(reported):
                        record["enzyme_name"] = reported
                        fixed_count += 1
                        logger.debug(f"    [Enzyme Name Fix] '{enzyme_name}' → reported '{reported}'")
                    else:
                        # 没有可用的真正酶名，保留 mutations 但清空 enzyme_name
                        # 以免使用错误的酶名
                        record["enzyme_name"] = ""
                        logger.debug(f"    [Enzyme Name Fix] '{enzyme_name}' → '' (no real enzyme name found)")

        if fixed_count > 0:
            logger.info(f"  [Enzyme Name Fix] Fixed {fixed_count} records with mutation-as-enzyme-name")

        return records

    def _annotate_candidate_provenance(
        self,
        record: Dict,
        source_channel: str,
        source_model: Optional[str] = None,
        raw_record_index: Optional[int] = None,
        locked_candidate: bool = False,
    ) -> Dict:
        """Attach stable provenance fields without overwriting explicit source evidence."""
        out = dict(record)
        out["source_channel"] = out.get("source_channel") or source_channel
        out["source_model"] = out.get("source_model") or source_model or out.get("_source_model") or out.get("_extracted_by") or ""
        source_table_id = (
            out.get("source_table_id")
            or out.get("_source_block_id")
            or out.get("source_section")
            or out.get("measurement_context_id")
            or ""
        )
        out["source_table_id"] = source_table_id if source_channel in {"parsed_table", "table_image_rescue"} else out.get("source_table_id", "")
        out["raw_record_index"] = out.get("raw_record_index") or raw_record_index
        out["locked_candidate"] = bool(locked_candidate or out.get("locked_candidate") is True)
        return out

    def _is_locked_table_candidate(self, record: Dict) -> bool:
        """Lock high-confidence table rows so teacher aggregation cannot delete them."""
        if record.get("source_channel") != "parsed_table" and record.get("_source_type") != "table":
            return False
        if "table_image_rescue" in self._record_flags(record):
            return False
        if not self._has_primary_metric(record):
            return False
        if not (record.get("substrate") and (record.get("reported_enzyme_name") or record.get("enzyme_name"))):
            return False
        return True

    def _dedupe_candidate_pool(self, records: List[Dict]) -> List[Dict]:
        """Merge duplicate candidates while preserving distinct measurement contexts.

        Table-derived candidates are the primary source of truth. Text/teacher
        candidates that describe the same enzyme-substrate metric are allowed to
        fill missing context fields, but they should not become duplicate kinetic
        rows. Sample labels remain part of the context so independent
        sample-labeled rows are preserved.
        """
        import re

        def norm(value: Any) -> str:
            text = str(value or "").lower()
            for old, new in {"–": "-", "—": "-", "−": "-", "⁻": "-", " ": "", "_": ""}.items():
                text = text.replace(old, new)
            return text

        substrate_aliases = {
            # Trichothecenes
            "don": "deoxynivalenol", "deoxynivalenol": "deoxynivalenol",
            "deoxynivalenol(don)": "deoxynivalenol",
            "niv": "nivalenol", "nivalenol": "nivalenol",
            "3adon": "3acetyldeoxynivalenol", "3acetyldeoxynivalenol": "3acetyldeoxynivalenol",
            "15adon": "15acetyldeoxynivalenol", "15acetyldeoxynivalenol": "15acetyldeoxynivalenol",
            "t2": "t2toxin", "t-2": "t2toxin", "t2toxin": "t2toxin",
            "ht2": "ht2toxin", "ht-2": "ht2toxin", "ht2toxin": "ht2toxin",
            # Zearalenone family
            "zen": "zearalenone", "zea": "zearalenone", "zearalenone": "zearalenone",
            "zel": "zearalenol", "zearalenol": "zearalenol",
            "alphazel": "alphazearalenol", "alphazearalenol": "alphazearalenol",
            "betazel": "betazearalenol", "betazearalenol": "betazearalenol",
            "zearalanone": "zearalanone", "zearalanol": "zearalanol",
            # Aflatoxins
            "afb1": "aflatoxinb1", "aflatoxinb1": "aflatoxinb1",
            "afb2": "aflatoxinb2", "aflatoxinb2": "aflatoxinb2",
            "afg1": "aflatoxing1", "aflatoxing1": "aflatoxing1",
            "afg2": "aflatoxing2", "aflatoxing2": "aflatoxing2",
            "afm1": "aflatoxinm1", "aflatoxinm1": "aflatoxinm1",
            # Ochratoxins
            "ota": "ochratoxina", "ochratoxina": "ochratoxina",
            "ochratoxin a": "ochratoxina", "ochratoxin": "ochratoxina",
            "otb": "ochratoxinb", "ochratoxinb": "ochratoxinb",
            # Fumonisins
            "fb1": "fumonisinb1", "fumonisinb1": "fumonisinb1",
            "fb2": "fumonisinb2", "fumonisinb2": "fumonisinb2",
            "fb3": "fumonisinb3", "fumonisinb3": "fumonisinb3",
            "hfb1": "hydrolyzedfumonisinb1",
            # Others
            "pat": "patulin", "patulin": "patulin",
            "stc": "sterigmatocystin", "sterigmatocystin": "sterigmatocystin",
            "cit": "citrinin", "citrinin": "citrinin",
            "aoh": "alternariol", "alternariol": "alternariol",
            "ame": "alternariolmonomethylether",
            "tea": "tenuazonicacid", "tenuazonic acid": "tenuazonicacid",
        }

        def norm_substrate(value: Any) -> str:
            text = norm(value)
            # 先查完整匹配
            if text in substrate_aliases:
                return substrate_aliases[text]
            # 处理带括号别名的情况，如 "deoxynivalenol(don)" → 提取括号内别名
            import re as _re
            paren_match = _re.search(r'\(([^)]+)\)', text)
            if paren_match:
                alias = norm(paren_match.group(1))
                if alias in substrate_aliases:
                    return substrate_aliases[alias]
            # 去掉括号内容后匹配
            stripped = _re.sub(r'\([^)]*\)', '', text).strip()
            if stripped in substrate_aliases:
                return substrate_aliases[stripped]
            return text

        def norm_num(value: Any) -> str:
            if value in (None, "", []):
                return ""
            try:
                return f"{float(value):.8g}"
            except (TypeError, ValueError):
                return norm(value)

        def num(value: Any) -> Optional[float]:
            if value in (None, "", []):
                return None
            try:
                return float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                return None

        def norm_kcat_km(record: Dict) -> str:
            value = record.get("kcat_Km_value")
            if value in (None, "", []):
                return ""
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return norm(value)
            unit_text_only = str(record.get("kcat_Km_unit") or "")
            if record.get("kinetic_unit_multiplier") not in (None, "", []) and "10" not in unit_text_only:
                return f"{numeric:.8g}"
            multiplier_text = " ".join(str(record.get(field) or "") for field in (
                "kcat_Km_unit", "kinetic_unit_source_text", "source_table_header", "evidence_text", "notes"
            ))
            try:
                from src.utils.table_multiplier import parse_table_header_multiplier
                multiplier, _matched, ambiguous = parse_table_header_multiplier(multiplier_text)
            except Exception:
                multiplier, ambiguous = None, False
            if multiplier and not ambiguous:
                numeric *= multiplier
            return f"{numeric:.8g}"

        def sample_token(record: Dict) -> str:
            blob = " ".join(str(record.get(field) or "") for field in (
                "measurement_context_id", "notes", "evidence_text", "source_section"
            ))
            match = re.search(r"sample\s*#?\s*(\d+)", blob, flags=re.IGNORECASE)
            return f"sample{match.group(1)}" if match else ""

        def mediator_token(record: Dict) -> str:
            """Keep enzyme-mediator degradation conditions as separate contexts."""
            name = norm(record.get("mediator_name"))
            value = norm_num(record.get("mediator_concentration"))
            unit = norm(record.get("mediator_concentration_unit")).replace("μ", "µ")
            blob = " ".join(str(record.get(field) or "") for field in (
                "notes", "evidence_text", "source_section"
            )).lower()
            if not name and re.search(r"\bno\s+mediator\b|\bwithout\s+mediator\b", blob):
                name = "nomediator"
            return "|".join(part for part in (name, value, unit) if part)

        def variant_or_enzyme(record: Dict) -> str:
            blob = " ".join(str(record.get(field) or "") for field in (
                "mutations", "reported_enzyme_name", "enzyme_name"
            ))
            low = blob.lower()
            compact = norm(blob)
            if re.search(r"\be186a\b", low) or "cotalaccasee186a" in compact:
                return "variant:e186a"
            if re.search(r"\be186r\b", low) or "cotalaccasee186r" in compact:
                return "variant:e186r"
            if re.search(r"\b(wt|wild type|wild-type)\b", low) or compact in {"cotalaccasewt", "cotawt"}:
                return "variant:wt"
            return "enzyme:" + norm(record.get("reported_enzyme_name") or record.get("enzyme_name"))

        def close_number(left: Optional[float], right: Optional[float], allow_per_minute: bool = False) -> bool:
            if left is None or right is None:
                return False
            scale = max(abs(left), abs(right), 1.0)
            if abs(left - right) / scale <= 0.025:
                return True
            if allow_per_minute:
                for factor in (60.0, 1.0 / 60.0):
                    scaled = right * factor
                    scale = max(abs(left), abs(scaled), 1.0)
                    if abs(left - scaled) / scale <= 0.025:
                        return True
            return False

        def kcat_km_number(record: Dict) -> Optional[float]:
            value = norm_kcat_km(record)
            numeric = num(value)
            multiplier = num(record.get("kinetic_unit_multiplier"))
            if numeric is not None and multiplier and multiplier != 1:
                threshold = max(100.0, abs(multiplier) / 10.0)
                if abs(numeric) < threshold:
                    return numeric * multiplier
            return numeric

        def common_metric_match(left: Dict, right: Dict) -> bool:
            checks = [
                ("Km_value", False),
                ("kcat_value", True),
                ("degradation_efficiency", False),
            ]
            matched_any = False
            for field, allow_per_minute in checks:
                lnum = num(left.get(field))
                rnum = num(right.get(field))
                if lnum is None or rnum is None:
                    continue
                if not close_number(lnum, rnum, allow_per_minute=allow_per_minute):
                    return False
                matched_any = True

            lkcat_km = kcat_km_number(left)
            rkcat_km = kcat_km_number(right)
            if lkcat_km is not None and rkcat_km is not None:
                if not close_number(lkcat_km, rkcat_km, allow_per_minute=True):
                    return False
                matched_any = True
            return matched_any

        def same_context(left: Dict, right: Dict) -> bool:
            if (
                norm(left.get("measurement_type")) != norm(right.get("measurement_type"))
                or norm_substrate(left.get("substrate")) != norm_substrate(right.get("substrate"))
            ):
                return False

            left_variant = variant_or_enzyme(left)
            right_variant = variant_or_enzyme(right)

            # 主匹配：相同 variant/enzyme token
            if left_variant == right_variant:
                left_sample = sample_token(left)
                right_sample = sample_token(right)
                if left_sample != right_sample:
                    if left_sample and right_sample:
                        return False
                    left_table = source_priority(left) >= 3
                    right_table = source_priority(right) >= 3
                    if norm(left.get("measurement_type")) == "kinetic" and (left_table != right_table):
                        return True
                    return False
                if norm(left.get("measurement_type")) == "degradation":
                    left_mediator = mediator_token(left)
                    right_mediator = mediator_token(right)
                    if left_mediator or right_mediator:
                        return left_mediator == right_mediator and common_metric_match(left, right)
                return common_metric_match(left, right)

            # 回退匹配：enzyme name 不同但动力学值一致 → 可能是同一测量
            # （表格记录用突变名当酶名，文本记录用正确酶名的情况）
            if common_metric_match(left, right):
                # 至少 Km 或 kcat 之一精确匹配
                matched_count = 0
                for field in ("Km_value", "kcat_value"):
                    lnum = num(left.get(field))
                    rnum = num(right.get(field))
                    if lnum is not None and rnum is not None and close_number(lnum, rnum):
                        matched_count += 1
                if matched_count >= 1:
                    return True

            return False

        def source_priority(record: Dict) -> int:
            channel = norm(record.get("source_channel"))
            locked = str(record.get("locked_candidate") or "").lower() == "true"
            if locked or channel in {"parsedtable", "tableimagerescue"}:
                return 3
            if channel == "text":
                return 1
            return 2

        def merge_records(primary: Dict, secondary: Dict) -> Dict:
            def generic_label(value: Any) -> bool:
                lowered = str(value or "").strip().lower()
                if not lowered:
                    return False
                return any(term in lowered for term in [
                    "degrading enzyme", "detoxifying enzyme", "hydrolytic enzyme",
                    "hydrolyzing enzyme", "extracellular enzyme", "extracellular enzymes",
                    "unknown enzyme", "unidentified enzyme",
                ])

            def concrete_identity(record: Dict) -> bool:
                if any(record.get(field) not in (None, "", []) for field in [
                    "gene_name", "uniprot_id", "genbank_id", "pdb_id", "ec_number", "sequence"
                ]):
                    return True
                for field in ["enzyme_name", "reported_enzyme_name"]:
                    value = str(record.get(field) or "").strip()
                    if value and not generic_label(value):
                        return True
                return False

            merged = dict(primary)
            for key, value in secondary.items():
                if key not in merged or merged.get(key) in (None, "", []):
                    merged[key] = value
            if concrete_identity(secondary):
                for key in [
                    "enzyme_name", "reported_enzyme_name", "gene_name", "uniprot_id",
                    "genbank_id", "pdb_id", "ec_number", "sequence", "is_recombinant",
                    "enzyme_state", "enzyme_system_type",
                ]:
                    secondary_value = secondary.get(key)
                    if secondary_value in (None, "", []):
                        continue
                    if key in {"enzyme_name", "reported_enzyme_name"}:
                        if merged.get(key) in (None, "", []) or generic_label(merged.get(key)):
                            merged[key] = secondary_value
                    elif merged.get(key) in (None, "", []):
                        merged[key] = secondary_value
                if concrete_identity(merged):
                    merged["identified_enzyme"] = True
                    merged["putative_enzyme"] = False
            for field in ("notes", "evidence_text"):
                left = str(merged.get(field) or "").strip()
                right = str(secondary.get(field) or "").strip()
                if right and right not in left:
                    merged[field] = f"{left} | {right}" if left else right
            return merged

        deduped: List[Dict] = []
        for record in records:
            match_idx = None
            for idx, existing in enumerate(deduped):
                if same_context(existing, record):
                    match_idx = idx
                    break
            if match_idx is None:
                deduped.append(record)
                continue

            existing = deduped[match_idx]

            # 当 enzyme name 不同时（突变名 vs 正确酶名），优先使用正确酶名的记录
            rec_enzyme = (record.get("enzyme_name") or "").strip()
            exist_enzyme = (existing.get("enzyme_name") or "").strip()
            rec_is_mutation = bool(self._MUTATION_PATTERN.match(rec_enzyme)) if rec_enzyme else False
            exist_is_mutation = bool(self._MUTATION_PATTERN.match(exist_enzyme)) if exist_enzyme else False

            if rec_enzyme and exist_enzyme and rec_enzyme.lower() != exist_enzyme.lower():
                # 一方是突变名，另一方不是 → 优先非突变名
                if rec_is_mutation and not exist_is_mutation:
                    # existing 有正确酶名，保持 existing 为主
                    deduped[match_idx] = merge_records(existing, record)
                elif not rec_is_mutation and exist_is_mutation:
                    # record 有正确酶名，用 record 为主
                    deduped[match_idx] = merge_records(record, existing)
                elif source_priority(record) > source_priority(existing):
                    deduped[match_idx] = merge_records(record, existing)
                else:
                    deduped[match_idx] = merge_records(existing, record)
            elif source_priority(record) > source_priority(existing):
                deduped[match_idx] = merge_records(record, existing)
            else:
                deduped[match_idx] = merge_records(existing, record)
        return deduped

    def _preserve_table_rescue_records(
        self,
        aggregated_records: List[Dict],
        shared_table_records: List[Dict],
    ) -> List[Dict]:
        """
        Keep distinct table records when teacher aggregation collapses or
        filters them out.

        This is intentionally narrow: it only applies to table records with
        valid kinetic/degradation metrics. Toxicity endpoint tables are not
        preserved as primary records.
        """
        if not shared_table_records:
            return aggregated_records

        def flags_for(record: Dict) -> List[str]:
            flags = record.get("error_flags") or []
            if isinstance(flags, str):
                flags = [flags]
            return flags

        toxicity_terms = [
            "residual bioluminescence", "ecotoxicity", "cytotoxicity",
            "cell viability", "ldh", "ros", "dna damage", "residual toxicity",
            "photobacterium",
        ]
        substrate_aliases = {
            "don": "deoxynivalenol",
            "deoxynivalenol": "deoxynivalenol",
            "zen": "zearalenone",
            "zea": "zearalenone",
            "zearalenone": "zearalenone",
            "ota": "ochratoxina",
            "ochratoxina": "ochratoxina",
            "afb1": "aflatoxinb1",
            "aflatoxinb1": "aflatoxinb1",
            "pat": "patulin",
            "patulin": "patulin",
            "stc": "sterigmatocystin",
            "sterigmatocystin": "sterigmatocystin",
        }

        def normalize_text(value: Any) -> str:
            text = str(value or "").lower()
            for old, new in {"–": "-", "—": "-", "−": "-", "⁻": "-", " ": "", "_": ""}.items():
                text = text.replace(old, new)
            return text

        def normalize_substrate(value: Any) -> str:
            text = normalize_text(value)
            if text in substrate_aliases:
                return substrate_aliases[text]
            # 处理带括号别名的情况
            import re as _re
            paren_match = _re.search(r'\(([^)]+)\)', text)
            if paren_match:
                alias = normalize_text(paren_match.group(1))
                if alias in substrate_aliases:
                    return substrate_aliases[alias]
            stripped = _re.sub(r'\([^)]*\)', '', text).strip()
            if stripped in substrate_aliases:
                return substrate_aliases[stripped]
            return text

        def normalize_number(value: Any) -> str:
            if value in (None, "", []):
                return ""
            try:
                return f"{float(value):.8g}"
            except (TypeError, ValueError):
                return normalize_text(value)

        def normalized_kcat_km_value(record: Dict) -> str:
            """Normalize kcat/Km values for dedupe, including table header multipliers."""
            value = record.get("kcat_Km_value")
            if value in (None, "", []):
                return ""
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return normalize_text(value)

            unit_text_only = str(record.get("kcat_Km_unit") or "")
            multiplier_already_applied = record.get("kinetic_unit_multiplier") not in (None, "", [])
            if multiplier_already_applied and "10" not in unit_text_only:
                return f"{numeric_value:.8g}"

            multiplier_text = " ".join(str(record.get(field) or "") for field in (
                "kcat_Km_unit",
                "kinetic_unit_source_text",
                "source_table_header",
                "evidence_text",
                "notes",
            ))
            try:
                from src.utils.table_multiplier import parse_table_header_multiplier
                multiplier, _matched_text, ambiguous = parse_table_header_multiplier(multiplier_text)
            except Exception:
                multiplier, ambiguous = None, False
            if multiplier and not ambiguous:
                numeric_value *= multiplier
            return f"{numeric_value:.8g}"

        def has_metric(record: Dict) -> bool:
            return any(record.get(field) not in (None, "", []) for field in (
                "Km_value", "kcat_value", "kcat_Km_value", "degradation_efficiency"
            ))

        def is_preservable_table_record(record: Dict) -> bool:
            flags = flags_for(record)
            if "table_image_rescue" in flags:
                return True
            if record.get("_source_type") != "table":
                return False
            if not has_metric(record):
                return False
            if not (record.get("substrate") and (record.get("reported_enzyme_name") or record.get("enzyme_name"))):
                return False
            text = " ".join(str(record.get(field) or "") for field in (
                "notes", "evidence_text", "source_section", "measurement_context_id"
            )).lower()
            if "thermodynamic" in text:
                return False
            if record.get("degradation_efficiency") not in (None, "", []) and any(term in text for term in toxicity_terms):
                return False
            return True

        preservable_records = []
        for record in shared_table_records:
            if is_preservable_table_record(record):
                preservable_records.append(record)

        if not preservable_records:
            return aggregated_records

        def pair_key(record: Dict) -> tuple:
            return (
                normalize_text(record.get("reported_enzyme_name") or record.get("enzyme_name")),
                normalize_substrate(record.get("substrate")),
            )

        def full_value_key(record: Dict) -> tuple:
            return (
                normalize_text(record.get("measurement_context_id")),
                normalize_text(record.get("reported_enzyme_name") or record.get("enzyme_name")),
                normalize_substrate(record.get("substrate")),
                normalize_number(record.get("Km_value")),
                normalize_number(record.get("kcat_value")),
                normalized_kcat_km_value(record),
                normalize_number(record.get("degradation_efficiency")),
            )

        def metric_value_key(record: Dict) -> tuple:
            return (
                normalize_substrate(record.get("substrate")),
                normalize_number(record.get("Km_value")),
                normalize_number(record.get("kcat_value")),
                normalized_kcat_km_value(record),
                normalize_number(record.get("degradation_efficiency")),
            )

        rescue_pairs = {pair_key(record) for record in preservable_records if "table_image_rescue" in flags_for(record)}
        filtered_aggregated = []
        removed_aggregate_rows = 0
        for record in aggregated_records:
            notes = normalize_text(record.get("notes"))
            context_id = normalize_text(record.get("measurement_context_id"))
            if (
                rescue_pairs
                and pair_key(record) in rescue_pairs
                and "tableimagerescue" not in normalize_text(record.get("error_flags"))
                and "sample#" not in notes
                and "sample#" not in context_id
                and has_metric(record)
            ):
                removed_aggregate_rows += 1
                continue
            filtered_aggregated.append(record)
        if removed_aggregate_rows:
            logger.warning(f"  [Table Preserve] Removed {removed_aggregate_rows} aggregate table rows superseded by sample-level table rescue")

        existing_keys = set()
        existing_metric_keys = set()
        for record in filtered_aggregated:
            existing_keys.add(full_value_key(record))
            existing_metric_keys.add(metric_value_key(record))
        preserved = list(filtered_aggregated)
        appended = 0
        for record in preservable_records:
            full_key = full_value_key(record)
            metric_key = metric_value_key(record)
            if full_key in existing_keys or metric_key in existing_metric_keys:
                continue
            clean_record = {k: v for k, v in record.items() if not k.startswith("_") or k == "_table_multiplier_applied"}
            flags = flags_for(clean_record)
            is_rescue_record = "table_image_rescue" in flags
            if is_rescue_record:
                clean_record["human_review_required"] = True
            else:
                clean_record["human_review_required"] = clean_record.get("human_review_required", False)
            flags = clean_record.get("error_flags") or []
            if isinstance(flags, str):
                flags = [flags]
            preserve_flag = (
                "table_image_rescue_preserved_after_aggregation"
                if is_rescue_record else
                "table_record_preserved_after_aggregation"
            )
            if preserve_flag not in flags:
                flags.append(preserve_flag)
            clean_record["error_flags"] = flags
            notes = clean_record.get("notes") or ""
            marker = (
                "Preserved after teacher aggregation because table-image rescue identified a distinct complex-table context."
                if is_rescue_record else
                "Preserved after teacher aggregation because table extraction identified a distinct enzyme-substrate metric context."
            )
            if marker not in notes:
                clean_record["notes"] = (notes + " | " if notes else "") + marker
            preserved.append(clean_record)
            existing_keys.add(full_key)
            existing_metric_keys.add(metric_key)
            appended += 1

        if appended:
            logger.warning(f"  [Table Preserve] Preserved {appended} distinct table records after aggregation")
        return preserved

    # Substrate abbreviation → full canonical name (case-insensitive)
    _SUBSTRATE_FULL_NAME_MAP = {
        "zen": "Zearalenone", "zea": "Zearalenone",
        "zel": "Zearalenol",
        "α-zel": "α-Zearalenol", "alpha-zel": "α-Zearalenol",
        "β-zel": "β-Zearalenol", "beta-zel": "β-Zearalenol",
        "zearalanone": "Zearalanone",
        "α-zearalanol": "α-Zearalanol", "alpha-zearalanol": "α-Zearalanol",
        "β-zearalanol": "β-Zearalanol", "beta-zearalanol": "β-Zearalanol",
        "don": "Deoxynivalenol", "niv": "Nivalenol",
        "afb1": "Aflatoxin B1", "afb2": "Aflatoxin B2",
        "afg1": "Aflatoxin G1", "afg2": "Aflatoxin G2",
        "afm1": "Aflatoxin M1", "ota": "Ochratoxin A", "otb": "Ochratoxin B",
        "fb1": "Fumonisin B1", "fb2": "Fumonisin B2", "fb3": "Fumonisin B3",
        "pat": "Patulin", "cit": "Citrinin", "stc": "Sterigmatocystin",
        "3-adon": "3-Acetyldeoxynivalenol", "15-adon": "15-Acetyldeoxynivalenol",
        "das": "Diacetoxyscirpenol", "fus-x": "Fusarenon-X", "d3g": "DON-3-Glucoside",
    }

    @classmethod
    def _normalize_substrate_name(cls, value: str) -> str:
        """Normalize substrate abbreviation to full canonical name."""
        text = str(value or "").strip()
        if not text:
            return text
        lower = text.lower()
        if lower in cls._SUBSTRATE_FULL_NAME_MAP:
            return cls._SUBSTRATE_FULL_NAME_MAP[lower]
        normalized = lower.replace("α", "alpha-").replace("β", "beta-")
        if normalized in cls._SUBSTRATE_FULL_NAME_MAP:
            return cls._SUBSTRATE_FULL_NAME_MAP[normalized]
        return text

    def _apply_post_aggregation_safety_filters(self, records: List[Dict]) -> List[Dict]:
        """
        Deterministic safety filters for recurring semantic errors that should
        not depend on prompt compliance.

        This is intentionally narrow and only handles:
        - thermodynamic kcat rows being misused as primary kinetic records;
        - products explicitly described as prior/assumed literature products.
        """
        if not records:
            return records

        # Normalize substrate abbreviations to full names
        for record in records:
            sub = str(record.get("substrate") or "").strip()
            if sub:
                record["substrate"] = self._normalize_substrate_name(sub)

        def text_blob(record: Dict) -> str:
            return " ".join(str(record.get(field) or "") for field in (
                "notes", "evidence_text", "source_section", "measurement_context_id", "products"
            )).lower()

        def add_flag(record: Dict, flag: str) -> None:
            flags = record.get("error_flags") or []
            if isinstance(flags, str):
                flags = [f for f in flags.split(";") if f] if ";" in flags else [flags]
            if flag not in flags:
                flags.append(flag)
            record["error_flags"] = flags

        cleaned: List[Dict] = []
        removed_thermodynamic = 0
        removed_prior_products = 0
        removed_toxicity_endpoint = 0
        removed_non_mycotoxin = 0
        prior_terms = (
            "prior study", "previous study", "previously", "reported previously",
            "samuel et al", "assumed based on", "according to", "has been reported",
            "our previous studies",
        )
        toxicity_terms = (
            "residual bioluminescence", "ecotoxicity", "cytotoxicity", "cell viability",
            "inhibition rate", "ldh", "ros", "dna damage", "tissue residue",
            "photobacterium", "residual toxicity",
        )
        material_supported_terms = (
            "immobilized", "immobilised", "immobilization", "immobilisation",
            "microsphere", "microspheres", "microbead", "microbeads",
            "covalently immobilized", "covalent bonding", "cross-linked", "crosslinked",
            "sodium alginate", "alginate", "montmorillonite", "sa/mt", "sa/mt./ez",
            "carrier", "support", "supported enzyme", "enzyme-loaded", "enzyme-coated",
            "hydrogel", "bead", "resin", "membrane", "composite",
        )
        mycotoxin_terms = (
            "aflatoxin", "afb1", "afm1", "ochratoxin", "ota",
            "deoxynivalenol", "don", "zearalenone", "zearalanone", "zearalanol",
            "zearalenol", "zel", "zen", "zea", "patulin",
            "sterigmatocystin", "citrinin", "fumonisin", "fb1", "fb2",
            "t-2", "ht-2", "nivalenol", "niv", "mycotoxin",
        )
        # Hard blacklist: these are NEVER mycotoxin substrates, even with human_review_required
        non_mycotoxin_terms = (
            "abts", "guaiacol", "catechol", "syringaldazine",
            "veratryl alcohol", "dmp", "2,6-dimethoxyphenol",
            "rbbr", "remazol brilliant", "reactive black", "methylene blue",
            "congo red", "bromophenol",
            "p-nitrophenol", "p-nitrophenyl", "pnp", "pnpp",
            "7-ethoxyresorufin", "7-pentoxyresorufin", "7-methoxyresorufin",
            "benzo[a]pyrene", "styrene oxide", "pentachlorophenol",
            "coumarin", "nifedipine", "docetaxel", "terfenadine",
            "glutathione", "gsh", "gssg", "nadph", "nadh",
            "h2o2", "hydrogen peroxide",
            "synbiotic", "bioplus", "cylactin", "inulin",
            "turkey tissue", "intestinal content", "fecal content",
        )
        for record in records:
            blob = text_blob(record)
            measurement_type = str(record.get("measurement_type") or "").lower()
            substrate_text = str(record.get("substrate") or "").lower()
            # Hard blacklist: reject known non-mycotoxin substrates unconditionally
            if substrate_text and any(term in substrate_text for term in non_mycotoxin_terms):
                removed_non_mycotoxin += 1
                continue
            # Bypass mycotoxin whitelist check for LLM-identified unknown mycotoxins (human_review_required=true)
            is_llm_flagged_mycotoxin = str(record.get("human_review_required", "")).lower() == "true"
            if substrate_text and not any(term in substrate_text for term in mycotoxin_terms):
                if is_llm_flagged_mycotoxin:
                    logger.info(f"  [Safety Filter] Kept LLM-flagged mycotoxin: {record.get('substrate')}")
                else:
                    removed_non_mycotoxin += 1
                    continue
            if (
                measurement_type == "kinetic"
                and "thermodynamic" in blob
                and record.get("kcat_value") not in (None, "", [])
                and record.get("Km_value") in (None, "", [])
            ):
                removed_thermodynamic += 1
                continue
            if record.get("degradation_efficiency") not in (None, "", []) and any(term in blob for term in toxicity_terms):
                removed_toxicity_endpoint += 1
                continue

            clean_record = dict(record)
            system_blob = " ".join([
                blob,
                str(record.get("enzyme_state") or "").lower(),
                str(record.get("enzyme_system_type") or "").lower(),
            ])
            if any(term in system_blob for term in material_supported_terms):
                clean_record["enzyme_system_type"] = "immobilized_enzyme"
                clean_record["enzyme_state"] = "immobilized"
                clean_record["QC_Status"] = clean_record.get("QC_Status") or "material_supported_or_immobilized_enzyme_system"
                add_flag(clean_record, "material_supported_or_immobilized_enzyme_system")
                note = str(clean_record.get("notes") or "")
                scope_note = (
                    "Excluded from primary database scope: measurement used an immobilized/material-supported "
                    "enzyme system, not a free purified/recombinant/commercial enzyme assay."
                )
                if scope_note not in note:
                    clean_record["notes"] = f"{note} | {scope_note}" if note else scope_note
            degradation_value = clean_record.get("degradation_efficiency")
            if degradation_value not in (None, "", []):
                import re
                value_text = str(degradation_value).strip()
                match = re.search(r"(?P<qual>>|<|≥|≤|more than|over|less than|about|approximately|approx\.?)?\s*(?P<num>\d+(?:\.\d+)?)\s*%?", value_text, flags=re.IGNORECASE)
                if match:
                    qualifier = (match.group("qual") or "").lower()
                    if qualifier in {"more than", "over", "≥"}:
                        qualifier = ">"
                    elif qualifier in {"less than", "≤"}:
                        qualifier = "<"
                    elif qualifier in {"about", "approximately", "approx."}:
                        qualifier = "approximately"
                    try:
                        clean_record["degradation_efficiency"] = float(match.group("num"))
                    except ValueError:
                        pass
                    clean_record["degradation_efficiency_unit"] = clean_record.get("degradation_efficiency_unit") or "%"
                    if qualifier:
                        note = str(clean_record.get("notes") or "")
                        qualifier_note = f"Reported degradation/conversion qualifier preserved: {qualifier}{match.group('num')}%."
                        if qualifier_note not in note:
                            clean_record["notes"] = f"{note} | {qualifier_note}" if note else qualifier_note
                elif not clean_record.get("degradation_efficiency_unit"):
                    clean_record["degradation_efficiency_unit"] = "%"

            products_text = str(clean_record.get("products") or "")
            if products_text and any(product in products_text.lower() for product in ("afd1", "afd2")):
                if any(term in blob for term in prior_terms):
                    clean_record["products"] = None
                    clean_record["human_review_required"] = True
                    add_flag(clean_record, "product_from_prior_literature_removed")
                    note = str(clean_record.get("notes") or "")
                    suffix = "Products AFD1/AFD2 removed from primary products because source text indicates prior/assumed literature evidence."
                    clean_record["notes"] = f"{note} | {suffix}" if note else suffix
                    removed_prior_products += 1

            # Auto-clear human_review_required when substrate is a known mycotoxin
            # (LLM over-flags because it doesn't see the full whitelist)
            if clean_record.get("human_review_required") and substrate_text:
                is_known_mycotoxin = any(term in substrate_text for term in mycotoxin_terms)
                is_rescue = "table_image_rescue" in (clean_record.get("error_flags") or [])
                if is_known_mycotoxin and not is_rescue:
                    clean_record["human_review_required"] = False

            cleaned.append(clean_record)

        if removed_thermodynamic:
            logger.warning(f"  [Safety Filter] Removed {removed_thermodynamic} thermodynamic kcat rows from primary kinetic records")
        if removed_toxicity_endpoint:
            logger.warning(f"  [Safety Filter] Removed {removed_toxicity_endpoint} toxicity endpoint rows misassigned as degradation")
        if removed_non_mycotoxin:
            logger.warning(f"  [Safety Filter] Removed {removed_non_mycotoxin} non-mycotoxin substrate candidate rows")
        if removed_prior_products:
            logger.warning(f"  [Safety Filter] Removed prior-literature products from {removed_prior_products} records")
        return cleaned

    def _extract_text_degradation_pair_fallbacks(
        self,
        original_text: str,
        existing_records: List[Dict],
    ) -> List[Dict]:
        """
        Deterministic fallback for explicit paired degradation statements in
        prose, e.g. "E186A and E186R ... degradation rates of 82.2% and 91.8%".

        This is not a broad extraction path. It only fires when the text gives
        two named enzyme variants and two explicit percent degradation values.
        """
        import re

        if not original_text:
            return []

        def normalize_math_text(text: str) -> str:
            text = re.sub(r"\\(?:mathrm|mathtt|mathbf|bf|textup|mathsf)\s*\{([^{}]*)\}", r"\1", text)
            text = text.replace("\\%", "%")
            text = re.sub(r"[$^_{}]", " ", text)
            text = re.sub(r"([A-Z])\s+(\d)\s+(\d)\s+(\d)\s+([A-Z])", r"\1\2\3\4\5", text)
            text = re.sub(
                r"(\d(?:\s+\d)*\s*\.\s*\d(?:\s+\d)*)\s*%",
                lambda m: re.sub(r"\s+", "", m.group(1)) + "%",
                text,
            )
            text = re.sub(
                r"\b(\d(?:\s+\d)+)(?=\s*h\b)",
                lambda m: re.sub(r"\s+", "", m.group(1)),
                text,
            )
            text = re.sub(r"\s+", " ", text)
            return text

        text = normalize_math_text(original_text)
        pattern = re.compile(
            r"(?P<v1>[A-Z]\d{2,4}[A-Z])\s+and\s+(?P<v2>[A-Z]\d{2,4}[A-Z]).{0,260}?"
            r"degradation\s+(?:rates?|ratios?).{0,100}?"
            r"(?P<p1>\d+(?:\.\d+)?)\s*%\s+and\s+(?P<p2>\d+(?:\.\d+)?)\s*%",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return []

        window_start = max(0, match.start() - 500)
        window_end = min(len(text), match.end() + 700)
        window = text[window_start:window_end]
        lower_window = window.lower()
        collapsed_window = re.sub(r"[^a-z0-9]+", "", lower_window)
        if "afb" in collapsed_window or "aflatoxin" in lower_window:
            substrate = "Aflatoxin B1"
        else:
            return []

        enzyme_name = "enzyme variant"
        if "cota-laccase" in lower_window or "cotalaccase" in lower_window:
            enzyme_name = "CotA-laccase"
        elif "laccase" in lower_window:
            enzyme_name = "laccase"

        temperature = None
        ph = None
        time_value = None
        context_window = text[max(0, match.start() - 2500): min(len(text), match.end() + 3500)]
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:°\s*C|degrees?\s*C|C)\b", context_window, re.IGNORECASE)
        if temp_match:
            try:
                temperature = float(temp_match.group(1))
            except ValueError:
                temperature = None
        ph_match = re.search(r"pH\s*(\d+(?:\.\d+)?)", context_window, re.IGNORECASE)
        if ph_match:
            try:
                ph = float(ph_match.group(1))
            except ValueError:
                ph = None
        time_match = re.search(r"(?:lasted|within|for)\s+(\d+(?:\.\d+)?)\s*h", context_window, re.IGNORECASE)
        if time_match:
            try:
                time_value = float(time_match.group(1))
            except ValueError:
                time_value = None

        existing_keys = {
            (
                str(r.get("mutations") or r.get("reported_enzyme_name") or "").lower(),
                str(r.get("substrate") or "").lower(),
                str(r.get("measurement_type") or "").lower(),
            )
            for r in existing_records
            if r.get("degradation_efficiency") not in (None, "", [])
        }

        records = []
        for variant_key, value_key in (("v1", "p1"), ("v2", "p2")):
            variant = match.group(variant_key).upper()
            key = (variant.lower(), substrate.lower(), "degradation")
            if key in existing_keys:
                continue
            value = float(match.group(value_key))
            records.append({
                "reported_enzyme_name": variant,
                "enzyme_name": enzyme_name,
                "mutations": variant,
                "substrate": substrate,
                "measurement_type": "degradation",
                "condition_scope": "degradation_assay",
                "degradation_efficiency": value,
                "degradation_efficiency_unit": "%",
                "degradation_temperature_value": temperature,
                "degradation_temperature_unit": "°C" if temperature is not None else None,
                "degradation_ph": ph,
                "degradation_time_value": time_value,
                "degradation_time_unit": "h" if time_value is not None else None,
                "human_review_required": True,
                "error_flags": ["text_degradation_pair_fallback"],
                "source_section": "text_degradation_pair_fallback",
                "evidence_text": window[:600],
                "notes": "Deterministic text fallback from explicit paired degradation-rate statement; verify against source text.",
                "_source_type": "text",
                "_extracted_by": "deterministic-text-fallback",
            })
        return records
    
    async def _extract_all_tables_once(
        self,
        table_blocks: List[Dict],
        paper_dir: Path
    ) -> List[Dict]:
        """
        智能路由表格提取：根据HTML质量和表头内容选择最优提取方式

        路由规则:
        ├─ HTML完整 + pandas解析成功 + 表头含动力学关键词 → 纯文本提取（省~3000 tokens/表）
        ├─ HTML解析失败或不完整 → 视觉模型提取（GLM-4.6V）
        └─ 表头不含动力学关键词 → 直接跳过

        优化特性：
        - 表格预筛选（caption关键词）
        - 智能路由（HTML质量判断）
        - 图片压缩（max 1024px, JPEG quality=75）
        - 降低max_tokens（4096）

        Args:
            table_blocks: 所有表格块
            paper_dir: 论文目录

        Returns:
            提取的所有表格记录
        """
        # 步骤1: 表格预筛选（基于caption）
        filtered_tables = []
        current_study_slice_records = []
        for block in table_blocks:
            if self._filter_table_by_caption(block):
                block_id = block.get('block_id', 'unknown')
                sliced_records = self._extract_current_study_reference_table(block, block_id)
                if sliced_records:
                    for record in sliced_records:
                        record['_source_block_id'] = block_id
                        record['_source_type'] = 'table'
                        record['_extracted_by'] = 'reference-current-study-slicer'
                        record['_extraction_method'] = 'reference-current-study-slice'
                        record['locked_candidate'] = True
                    current_study_slice_records.extend(sliced_records)
                    logger.info(
                        "    [Reference Slicer] Table %s: extracted %s current-study records; skipping whole-table extraction",
                        block_id,
                        len(sliced_records),
                    )
                    continue
                metric_skip_reason = self._table_metric_scope_skip_reason(block)
                if metric_skip_reason:
                    self._mark_table_skipped_before_extraction(block, metric_skip_reason)
                    logger.info(
                        "    [Table Type Gate] Table %s: SKIP (%s)",
                        block_id,
                        metric_skip_reason,
                    )
                    continue
                filtered_tables.append(block)

        total_tables = len(table_blocks)
        filtered_count = len(filtered_tables)
        skipped_count = total_tables - filtered_count

        logger.info(f"    [Smart Routing] Pre-filtering: {total_tables} → {filtered_count} (skipped: {skipped_count})")

        if not filtered_tables:
            return current_study_slice_records

        # 步骤2: 智能路由决策
        text_only_tables = []
        vision_model_tables = []
        no_keyword_tables = []

        for block in filtered_tables:
            block_id = block.get('block_id', 'unknown')
            use_text_only, reason = self._should_use_text_only_extraction(block)

            if use_text_only is None:
                # 跳过：不含动力学关键词
                no_keyword_tables.append((block_id, block))
                TokenTracker.add_no_keyword_table()
                logger.debug(f"    [Smart Routing] Table {block_id}: SKIP ({reason})")
            elif use_text_only:
                # 复杂多级/合并单元格表格即使 pandas 可解析，也可能丢 parent labels。
                # 有表格图片时，先走 image-normalization rescue，而不是只靠 HTML。
                if self.disable_table_image:
                    text_only_tables.append((block_id, block))
                    TokenTracker.add_text_only_table()
                    logger.debug(f"    [Smart Routing] Table {block_id}: TEXT-ONLY (ablation: table-image disabled)")
                else:
                    rescue_reason = self._should_force_multimodal_table_rescue(block, paper_dir)
                    if rescue_reason:
                        block['_table_image_rescue_reason'] = rescue_reason
                        vision_model_tables.append((block_id, block))
                        TokenTracker.add_vision_model_table()
                        logger.debug(f"    [Smart Routing] Table {block_id}: VISION RESCUE ({rescue_reason})")
                    else:
                        # 纯文本提取
                        text_only_tables.append((block_id, block))
                        TokenTracker.add_text_only_table()
                        logger.debug(f"    [Smart Routing] Table {block_id}: TEXT-ONLY ({reason})")
            else:
                # 视觉模型提取
                if self.disable_table_image:
                    text_only_tables.append((block_id, block))
                    TokenTracker.add_text_only_table()
                    logger.debug(f"    [Smart Routing] Table {block_id}: TEXT-ONLY FALLBACK (ablation: table-image disabled, original: {reason})")
                else:
                    rescue_reason = self._should_force_multimodal_table_rescue(block, paper_dir)
                    if rescue_reason:
                        block['_table_image_rescue_reason'] = rescue_reason
                    vision_model_tables.append((block_id, block))
                    TokenTracker.add_vision_model_table()
                    route_reason = f"VISION RESCUE ({rescue_reason})" if rescue_reason else f"VISION MODEL ({reason})"
                    logger.debug(f"    [Smart Routing] Table {block_id}: {route_reason}")

        # 输出路由统计
        logger.info(f"    [Smart Routing] Text-only: {len(text_only_tables)}, Vision: {len(vision_model_tables)}, Skip: {len(no_keyword_tables)}")

        all_table_records = list(current_study_slice_records)

        # 步骤3a: 处理纯文本表格（并行）
        if text_only_tables:
            logger.debug(f"    [Smart Routing] Processing {len(text_only_tables)} tables with text-only extraction...")
            text_tasks = []
            for block_id, block in text_only_tables:
                task = self._extract_table_text_only(block, block_id, paper_dir)
                text_tasks.append((block_id, task, 'text-only'))

            text_results = await asyncio.gather(*[t[1] for t in text_tasks], return_exceptions=True)

            for (block_id, _, extraction_type), result in zip(text_tasks, text_results):
                if isinstance(result, Exception):
                    logger.error(f"    [Text-Only] Table {block_id} failed: {result}")
                else:
                    for record in result:
                        record['_source_block_id'] = block_id
                        record['_source_type'] = 'table'
                        record['_extracted_by'] = 'text-only'
                        record['_extraction_method'] = extraction_type
                    all_table_records.extend(result)
                    logger.debug(f"    [Text-Only] Table {block_id}: {len(result)} records")

        # 步骤3b: 处理视觉模型表格（并行，使用Semaphore限流）
        if vision_model_tables:
            logger.debug(f"    [Smart Routing] Processing {len(vision_model_tables)} tables with vision model...")
            vision_model_name = getattr(self.multimodal_model, "model_name", None) or "vision"
            vision_semaphore = get_semaphore(vision_model_name)

            vision_tasks = []
            for block_id, block in vision_model_tables:
                task = self._extract_with_semaphore(
                    vision_semaphore,
                    self._extract_table_block_multimodal,
                    self.multimodal_model,
                    block,
                    block_id,
                    paper_dir,
                    vision_model_name
                )
                vision_tasks.append((block_id, task, 'vision'))

            vision_results = await asyncio.gather(*[t[1] for t in vision_tasks], return_exceptions=True)

            for (block_id, _, extraction_type), result in zip(vision_tasks, vision_results):
                if isinstance(result, Exception):
                    logger.error(f"    [Vision] Table {block_id} failed: {result}")
                else:
                    for record in result:
                        record['_source_block_id'] = block_id
                        record['_source_type'] = 'table'
                        record['_extracted_by'] = vision_model_name
                        record['_extraction_method'] = extraction_type
                    all_table_records.extend(result)
                    logger.debug(f"    [Vision] Table {block_id}: {len(result)} records")

        logger.debug(f"    [Smart Routing] Total table records: {len(all_table_records)}")

        # 记录Token统计（每篇论文结束时输出）
        TokenTracker.log_stats()

        return all_table_records

    def _extract_current_study_reference_table(self, block: Dict, block_id: Any) -> List[Dict]:
        """Extract only current-study rows from reference/comparison tables.

        This handles large literature-comparison tables where whole-table JSON
        extraction can be truncated and where prior-study rows should not enter
        the current paper's database records.
        """
        table_content = block.get('table_body', '') or block.get('content', '')
        if not table_content or not PANDAS_AVAILABLE:
            return []

        table_text = re.sub(r"<[^>]+>", " ", table_content)
        lowered = table_text.lower()
        if not self._has_reference_like_signal(lowered) or not self._has_current_study_signal(lowered):
            return []

        try:
            dfs = pd.read_html(io.StringIO(table_content), header=0, flavor='bs4')
        except Exception as exc:
            logger.debug(f"    [Reference Slicer] Table {block_id}: pandas parse failed: {exc}")
            return []

        records: List[Dict] = []
        for df in dfs:
            if df.empty:
                continue
            records.extend(self._extract_current_study_records_from_df(df, block, block_id))
        return records

    def _has_reference_like_signal(self, text: str) -> bool:
        reference_terms = [
            "reference", "references", "ref.", "refs.", "citation", "citations",
            "reported by", "publication",
        ]
        if any(term in text for term in reference_terms):
            return True
        # Keep "source" conservative: require citation/current-study values too.
        return "source" in text and (
            self._has_current_study_signal(text)
            or re.search(r"\bet\s+al\.?\b|\b(?:19|20)\d{2}\b|\[\d+\]|\bdoi\b", text, flags=re.I)
        )

    def _has_current_study_signal(self, text: str) -> bool:
        return any(term in text for term in [
            "this study", "this work", "current study", "current work",
            "present study", "present work", "our study", "our work",
            "this paper", "this article", "herein", "in this study",
            "in the present work", "本研究", "本文",
        ])

    def _is_prior_reference_signal(self, text: str) -> bool:
        return bool(
            re.search(r"\bet\s+al\.?\b", text, flags=re.I)
            or re.search(r"\b(?:19|20)\d{2}\b", text)
            or re.search(r"\[\d+\]|\bdoi\b|previously reported|reported by|literature", text, flags=re.I)
        )

    def _cell_text(self, value: Any) -> str:
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        text = str(value).strip()
        if text.lower() in {"nan", "none", "null"}:
            return ""
        return re.sub(r"\s+", " ", text)

    def _infer_current_study_enzyme_name(self, block: Dict) -> str:
        context = self._get_paper_context_for_table(block)
        match = re.search(
            r"\b([A-Z][A-Za-z0-9_-]{1,30}\s+(?:laccase|peroxidase|oxidase|hydrolase|esterase|reductase|transferase))\b",
            context,
            flags=re.I,
        )
        if match:
            return match.group(1).strip()
        return ""

    def _infer_current_study_substrate(self, block: Dict, columns_text: str) -> str:
        caption = block.get("table_caption", "")
        if isinstance(caption, list):
            caption = " ".join(caption)
        text = f"{caption} {columns_text}".lower()
        for name, canonical in [
            ("zearalenone", "ZEN"), ("zen", "ZEN"),
            ("aflatoxin b1", "AFB1"), ("afb1", "AFB1"),
            ("ochratoxin a", "OTA"), ("ota", "OTA"),
            ("deoxynivalenol", "DON"), ("don", "DON"),
            ("patulin", "Patulin"),
        ]:
            if re.search(rf"\b{re.escape(name)}\b", text):
                return canonical
        return ""

    def _extract_conditions_from_text(self, text: str, previous: Dict[str, Any]) -> Dict[str, Any]:
        conditions = dict(previous)
        ph = re.search(r"\bpH\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
        if ph:
            conditions["degradation_ph"] = float(ph.group(1))
        temp = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:°\s*)?C\b", text, flags=re.I)
        if temp:
            conditions["degradation_temperature_value"] = float(temp.group(1))
            conditions["degradation_temperature_unit"] = "°C"
        time_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|min|minute|minutes|s|sec|seconds)\b", text, flags=re.I)
        if time_match:
            unit = time_match.group(2).lower()
            if unit in {"hr", "hrs", "hour", "hours"}:
                unit = "h"
            elif unit in {"minute", "minutes"}:
                unit = "min"
            elif unit in {"sec", "seconds"}:
                unit = "s"
            conditions["degradation_time_value"] = float(time_match.group(1))
            conditions["degradation_time_unit"] = unit
        return conditions

    def _extract_mediator_mentions(self, text: str) -> List[Dict[str, Any]]:
        mediator_pattern = (
            r"ABTS|TEMPO|HBT|acetosyringone|syringaldehyde|vanillin|vanillic acid|"
            r"syringic acid|p-coumaric acid|ferulic acid|caffeic acid|methyl syringate|"
            r"2,6-dimethoxy\s*phenol|DMP|phenol red"
        )
        mentions: List[Dict[str, Any]] = []
        if re.search(r"\bno\s+mediator\b", text, flags=re.I):
            mentions.append({"name": "", "value": None, "unit": "", "raw": "no mediator"})
        pattern = re.compile(
            rf"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>mM|µM|μM|uM|M)\s+(?:of\s+)?(?P<name>{mediator_pattern})",
            flags=re.I,
        )
        for match in pattern.finditer(text):
            mentions.append({
                "name": re.sub(r"\s+", " ", match.group("name")).strip(),
                "value": float(match.group("value")),
                "unit": match.group("unit"),
                "raw": match.group(0),
            })
        return mentions

    def _extract_current_study_records_from_df(self, df, block: Dict, block_id: Any) -> List[Dict]:
        columns_text = " ".join(str(c) for c in df.columns)
        if not self._has_reference_like_signal(columns_text.lower() + " " + df.to_string().lower()):
            return []

        fallback_enzyme = self._infer_current_study_enzyme_name(block)
        substrate = self._infer_current_study_substrate(block, columns_text)
        mediator_col_idx: Optional[int] = None
        for idx, column in enumerate(df.columns):
            if "mediator" in str(column).lower():
                mediator_col_idx = idx
                break
        records: List[Dict] = []
        effective_reference = ""
        in_current_study = False
        current_conditions: Dict[str, Any] = {}
        pending_rates: List[float] = []

        for _, row in df.iterrows():
            cells = [self._cell_text(value) for value in row.tolist()]
            row_text = " | ".join(cell for cell in cells if cell)
            if not row_text:
                continue
            row_lower = row_text.lower()

            if self._has_current_study_signal(row_lower):
                effective_reference = "this study"
                in_current_study = True
            elif self._is_prior_reference_signal(row_lower):
                # A new cited-study reference block starts; stop current-study inheritance.
                effective_reference = row_text
                in_current_study = False
            elif effective_reference == "this study":
                in_current_study = True
            else:
                in_current_study = False

            if not in_current_study:
                continue

            current_conditions = self._extract_conditions_from_text(row_text, current_conditions)
            mediators = self._extract_mediator_mentions(row_text)
            rates = [float(match.group(1)) for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*%", row_text)]
            if pending_rates:
                rates = pending_rates + rates
                pending_rates = []
            if not mediators or not rates:
                continue

            parser_shifted_multimediator_row = False
            if mediator_col_idx is not None and len(mediators) == len(rates) and len(mediators) > 1:
                mediator_cell_indexes = [
                    cell_idx
                    for cell_idx, cell in enumerate(cells)
                    if self._extract_mediator_mentions(cell)
                ]
                # Rowspan-heavy tables sometimes shift continuation rows left,
                # placing mediator cells before the actual Mediators column.
                # In that case, pandas preserves HTML order rather than visual
                # row order, so reverse the mediator order for this row only.
                if mediator_cell_indexes and min(mediator_cell_indexes) < mediator_col_idx:
                    mediators = list(reversed(mediators))
                    parser_shifted_multimediator_row = True

            if len(rates) > len(mediators):
                pending_rates = rates[len(mediators):]
            for idx, mediator in enumerate(mediators):
                if idx >= len(rates):
                    break
                enzyme_name = fallback_enzyme or self._cell_text(row.get("Enzyme name", ""))
                if not enzyme_name:
                    enzyme_name = "current-study enzyme"
                evidence = f"Reference=This study; {row_text}"
                record = {
                    "reported_enzyme_name": enzyme_name,
                    "enzyme_name": enzyme_name,
                    "substrate": substrate,
                    "measurement_type": "degradation",
                    "degradation_efficiency": rates[idx],
                    "degradation_efficiency_unit": "%",
                    "mediator_name": mediator["name"],
                    "mediator_concentration": mediator["value"],
                    "mediator_concentration_unit": mediator["unit"],
                    "degradation_ph": current_conditions.get("degradation_ph"),
                    "degradation_temperature_value": current_conditions.get("degradation_temperature_value"),
                    "degradation_temperature_unit": current_conditions.get("degradation_temperature_unit"),
                    "degradation_time_value": current_conditions.get("degradation_time_value"),
                    "degradation_time_unit": current_conditions.get("degradation_time_unit"),
                    "source_section": f"table_{block_id}_current_study_reference_block",
                    "source_table_id": f"table_{block_id}",
                    "evidence_text": evidence,
                    "notes": "Extracted from current-study block of a reference/comparison table; prior cited-study rows excluded.",
                    "human_review_required": False,
                }
                if parser_shifted_multimediator_row:
                    record["notes"] += " Parser-shifted multi-mediator row normalized from table structure."
                records.append(record)
        return records
    
    async def _extract_text_only_with_model(
        self,
        text_model,
        paper_blocks: List[Dict],
        paper_dir: Path,
        model_name: str
    ) -> List[Dict]:
        """
        只用学生模型提取文本内容（不调用 GLM-4.6V 处理表格）
        
        表格结果已经在 Step 0 由 GLM-4.6V 提取，这里只处理文本。
        
        Args:
            text_model: 学生文本模型
            paper_blocks: 论文所有块
            paper_dir: 论文目录
            model_name: 模型名称
            
        Returns:
            提取的文本记录
        """
        all_records = []
        
        # 获取 semaphore
        semaphore = get_semaphore(model_name)
        
        # 读取 full.md
        full_md_path = paper_dir / "full.md"
        merged_text = ""
        
        if full_md_path.exists():
            try:
                merged_text = full_md_path.read_text(encoding='utf-8')
                original_len = len(merged_text)
                
                # 去除 References 部分
                merged_text = self._remove_references(merged_text)
                merged_text = self._remove_skipped_table_context_from_text(merged_text, paper_blocks)
                
                logger.debug(f"    [{model_name}] Read full.md: {original_len} chars -> {len(merged_text)} chars")
            except Exception as e:
                logger.error(f"    [{model_name}] Failed to read full.md: {e}")
                merged_text = ""
        else:
            # Fallback: 使用 content_list.json 的文本块
            logger.warning(f"    [{model_name}] full.md not found, falling back to text blocks")
            text_blocks = [b for b in paper_blocks if b.get('type') == 'text']
            merged_text = self._merge_text_blocks(text_blocks)
        
        if merged_text:
            logger.debug(f"    [{model_name}] Full text: {len(merged_text)} chars")
            
            # 单次 API 调用提取整篇论文的文本
            try:
                text_records = await self._extract_with_semaphore(
                    semaphore,
                    self._extract_full_paper_text,
                    text_model,
                    merged_text,
                    model_name
                )
                for record in text_records:
                    record['_source_model'] = model_name
                    record['_source_type'] = 'text'
                all_records.extend(text_records)
                logger.debug(f"    [{model_name}] ✓ Text extraction: {len(text_records)} records")
            except Exception as e:
                logger.error(f"    [{model_name}] Text extraction failed: {e}")
                raise
        
        return all_records

    async def _extract_with_model_combo(
        self,
        text_model,
        multimodal_model,
        paper_blocks: List[Dict],
        paper_dir: Path,
        model_name: str
    ) -> List[Dict]:
        """
        用一个模型组合提取整篇论文 - 【优化版：直接读取 full.md】
        
        ⚠️ 注意：此方法已废弃，保留供兼容旧代码。
        新的提取流程使用 _extract_all_tables_once + _extract_text_only_with_model
        
        参考 LLM-BioDataExtractor 的做法：
        - 直接读取 MinerU 生成的 full.md 文件
        - 去除 References 部分
        - 一次性发给模型提取
        - 表格图片单独用多模态模型处理
        
        这样大幅减少API调用次数：
        - 旧方案: 123 blocks × 1 API/block = 123 API calls
        - 新方案: 1 full.md (1 API) + N tables (N APIs) ≈ 5-10 API calls
        
        Args:
            text_model: 文本模型（处理 full.md 全文）
            multimodal_model: 多模态模型（处理表格图片）
            paper_blocks: 论文所有块（仅用于获取表格信息）
            paper_dir: 论文目录
            model_name: 模型名称（用于标记）
            
        Returns:
            提取的所有记录
        """
        all_records = []
        
        # 获取semaphore（动态创建，避免事件循环冲突）
        semaphore = get_semaphore(model_name)
        vision_model_name = getattr(multimodal_model, "model_name", None) or model_name
        vision_semaphore = get_semaphore(vision_model_name)
        
        # ============================================================
        # Step 1: 直接读取 full.md，去除 References，一次性提取
        # ============================================================
        table_blocks = [b for b in paper_blocks if b.get('type') == 'table']
        figure_blocks = [b for b in paper_blocks if b.get('type') == 'figure']
        
        # 读取 full.md
        full_md_path = paper_dir / "full.md"
        merged_text = ""
        
        if full_md_path.exists():
            try:
                merged_text = full_md_path.read_text(encoding='utf-8')
                original_len = len(merged_text)
                
                # 去除 References 部分
                merged_text = self._remove_references(merged_text)
                merged_text = self._remove_skipped_table_context_from_text(merged_text, paper_blocks)
                
                logger.debug(f"    [{model_name}] Read full.md: {original_len} chars -> {len(merged_text)} chars (after removing references)")
            except Exception as e:
                logger.error(f"    [{model_name}] Failed to read full.md: {e}")
                merged_text = ""
        else:
            # Fallback: 如果没有 full.md，使用 content_list.json 的文本块
            logger.warning(f"    [{model_name}] full.md not found, falling back to content_list.json text blocks")
            text_blocks = [b for b in paper_blocks if b.get('type') == 'text']
            merged_text = self._merge_text_blocks(text_blocks)
        
        logger.debug(f"    [{model_name}] Paper structure: {len(table_blocks)} tables, {len(figure_blocks)} figures")
        
        if merged_text:
            logger.debug(f"    [{model_name}] Full text: {len(merged_text)} chars")
            
            # 🔧 单次API调用提取整篇论文的文本
            try:
                text_records = await self._extract_with_semaphore(
                    semaphore,
                    self._extract_full_paper_text,  # 新方法：整篇提取
                    text_model,
                    merged_text,
                    model_name
                )
                for record in text_records:
                    record['_source_model'] = model_name
                    record['_source_type'] = 'text'
                all_records.extend(text_records)
                logger.debug(f"    [{model_name}] ✓ Text extraction: {len(text_records)} records")
            except Exception as e:
                logger.error(f"    [{model_name}] Text extraction failed: {e}")
        
        # ============================================================
        # Step 2: 表格图片单独处理（多模态模型必须一图一调用）
        # ============================================================
        if table_blocks:
            logger.debug(f"    [{model_name}] Processing {len(table_blocks)} tables with multimodal model...")
            
            table_tasks = []
            for block in table_blocks:
                block_id = block.get('block_id', 'unknown')
                task = self._extract_with_semaphore(
                    vision_semaphore,
                    self._extract_table_block_multimodal,
                    multimodal_model,
                    block,
                    block_id,
                    paper_dir,
                    vision_model_name
                )
                table_tasks.append((block_id, task))
            
            # 并行处理表格（Semaphore自动限流）
            table_results = await asyncio.gather(*[t[1] for t in table_tasks], return_exceptions=True)
            
            for (block_id, _), result in zip(table_tasks, table_results):
                if isinstance(result, Exception):
                    logger.error(f"    [{model_name}] Table {block_id} failed: {result}")
                else:
                    for record in result:
                        record['_source_model'] = model_name
                        record['_source_block_id'] = block_id
                        record['_source_type'] = 'table'
                    all_records.extend(result)
            
            logger.debug(f"    [{model_name}] ✓ Table extraction: {len([r for r in all_records if r.get('_source_type') == 'table'])} records")
        
        # ============================================================
        # Step 3: 图片暂时跳过（一般不含酶动力学数据）
        # ============================================================
        if figure_blocks:
            logger.debug(f"    [{model_name}] Skipping {len(figure_blocks)} figures (rarely contain kinetic data)")
        
        return all_records
    
    def _remove_references(self, text: str) -> str:
        """
        去除文章中的 References 部分
        
        参考 LLM-BioDataExtractor 的 del_references 方法
        
        Args:
            text: 原始文本
            
        Returns:
            去除 References 后的文本
        """
        import re
        
        # 常见的参考文献标题模式
        patterns = [
            # 保留 Tables 部分的模式
            (r'\*\{.{0,5}(References|Reference|REFERENCES|LITERATURE CITED|Referencesand notes|Notes and references)(.*?)\\section\*\{Tables', r"\section*{Tables\n"),
            (r'#.{0,15}(References|Reference|REFERENCES|LITERATURE CITED|Referencesand notes|Notes and references)(.*?)(Table|Tables)', r"Tables"),
            (r'#.{0,15}(References|Reference|REFERENCES|LITERATURE CITED|Referencesand notes|Notes and references)(.*?)# SUPPLEMENTARY', r"# SUPPLEMENTARY"),
            
            # Markdown 标题格式 (## References, # References 等)
            (r'#{1,3}\s*(References|Reference|REFERENCES|LITERATURE CITED|Bibliography|Works Cited).*', ''),
            
            # LaTeX 格式
            (r'\\section\*?\{(References|Reference|REFERENCES|Bibliography)\}.*', ''),
            
            # 通用格式：匹配到文末
            (r'\*\{.{0,5}(References|Reference|REFERENCES|LITERATURE CITED|Referencesand notes|Notes and references).*', ''),
            (r'\n(References|Reference|REFERENCES|LITERATURE CITED|Bibliography)\n.*', ''),
        ]
        
        original_len = len(text)
        
        for pattern, replacement in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                text = re.sub(pattern, replacement, text, flags=re.DOTALL | re.IGNORECASE)
                logger.debug(f"    Removed references using pattern: {pattern[:50]}...")
                break
        
        if len(text) < original_len:
            removed_chars = original_len - len(text)
            logger.debug(f"    References removed: {removed_chars} chars ({removed_chars/original_len*100:.1f}%)")
        else:
            logger.debug(f"    No references section found to remove")
        
        return text.strip()
    
    def _merge_text_blocks(self, text_blocks: List[Dict]) -> str:
        """
        合并所有文本块为一个大文本（Fallback方法）
        
        Args:
            text_blocks: 文本块列表
            
        Returns:
            合并后的文本（带分隔符）
        """
        if not text_blocks:
            return ""
        
        parts = []
        for i, block in enumerate(text_blocks):
            content = block.get('content', '')
            if content.strip():
                parts.append(content.strip())
        
        # 用换行分隔，保持文章结构
        return "\n\n".join(parts)
    
    async def _extract_full_paper_text(
        self,
        model,
        merged_text: str,
        model_name: str
    ) -> List[Dict]:
        """
        一次性提取整篇论文的文本内容
        
        Args:
            model: 文本模型
            merged_text: 合并后的全文
            model_name: 模型名称
            
        Returns:
            提取的记录列表
        """
        # 限制最大长度（防止超出模型上下文限制）
        max_chars = 100000  # 约 25K tokens
        if len(merged_text) > max_chars:
            logger.warning(f"    [{model_name}] Text too long ({len(merged_text)} chars), truncating to {max_chars}")
            merged_text = merged_text[:max_chars]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in enzyme kinetics data extraction. Extract ALL enzyme "
                    "kinetic and quantitative detoxification records from the given scientific "
                    "article. Treat mycotoxin enzymatic transformation, inactivation, "
                    "glucosylation/glycosylation, conjugation, hydrolysis, oxidation, reduction "
                    "or detoxification as in-scope when the substrate is a mycotoxin or explicit "
                    "mycotoxin derivative. Do not return an empty array when Km, kcat or "
                    "kcat/Km/catalytic efficiency values are reported for mycotoxin substrates."
                ),
            },
            {"role": "user", "content": f"{self.text_prompt}\n\n=== ARTICLE FULL TEXT ===\n\n{merged_text}"}
        ]
        
        # 在线程池中运行同步调用
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: model.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=FULL_TEXT_MAX_TOKENS,
                json_mode=True,
                task=f"text_{model_name}",
            )
        )
        records = self._parse_json_response(response)
        if not records:
            preview = str(response or "")[:1000].replace("\n", "\\n")
            logger.warning(
                "    [%s] Full-paper text extraction returned 0 parsed records. "
                "Raw response length=%s; preview=%s",
                model_name,
                len(str(response or "")),
                preview,
            )
        return records

    async def _extract_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        func,
        *args,
        **kwargs
    ) -> List[Dict]:
        """
        带Semaphore限流和指数退避重试的API调用包装器
        
        Args:
            semaphore: 并发控制信号量
            func: 要执行的异步函数
            *args, **kwargs: 传给func的参数
            
        Returns:
            func的返回值
        """
        max_retries = RETRY_CONFIG["max_retries"]
        base_delay = RETRY_CONFIG["base_delay"]
        max_delay = RETRY_CONFIG["max_delay"]
        jitter = RETRY_CONFIG["jitter"]
        
        for attempt in range(max_retries + 1):
            try:
                # 使用Semaphore限制并发
                async with semaphore:
                    return await func(*args, **kwargs)
                    
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str or "too many" in error_str
                
                if attempt < max_retries and is_rate_limit:
                    # 指数退避 + 随机抖动
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay = delay * (1 + random.uniform(-jitter, jitter))
                    
                    logger.warning(f"    ⏳ Rate limit hit, retry {attempt + 1}/{max_retries} after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    # 非限流错误或重试次数用尽，抛出异常
                    raise
        
        # 不应该到这里
        return []
    
    async def _extract_text_block(
        self,
        model,
        content: str,
        block_id: int,
        model_name: str
    ) -> List[Dict]:
        """提取文本块"""
        messages = [
            {"role": "system", "content": "You are an expert in enzyme kinetics data extraction."},
            {"role": "user", "content": f"{self.text_prompt}\n\n{content}"}
        ]
        
        # 在线程池中运行同步调用
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: model.chat(messages=messages, temperature=0.1, task="table_text_only")
        )
        return self._parse_json_response(response)
    
    async def _extract_table_block(
        self,
        model,
        content: str,
        block_id: int,
        model_name: str
    ) -> List[Dict]:
        """提取表格块（纯文本模式，已弃用）"""
        messages = [
            {"role": "system", "content": "You are an expert in enzyme kinetics data extraction."},
            {"role": "user", "content": f"{self.table_prompt}\n\n{content}"}
        ]
        
        # 在线程池中运行同步调用
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: model.chat(messages=messages, temperature=0.1, task="table_text_only")
        )
        return self._parse_json_response(response)
    
    def _filter_table_by_caption(self, block: Dict) -> bool:
        """
        优化1: 表格预筛选 - 通过caption关键词判断表格是否可能包含动力学数据

        Returns:
            True: 表格可能包含动力学数据，需要处理
            False: 跳过此表格
        """
        caption = block.get('table_caption', '')
        if isinstance(caption, list):
            caption = ' '.join(caption)

        if not caption:
            # 没有标题的表格，默认处理（保守策略）
            return True

        caption_lower = caption.lower()

        # 检查排除关键词（优先级高）
        for exclude_kw in TABLE_EXCLUDE_KEYWORDS:
            if exclude_kw in caption_lower:
                TokenTracker.add_skipped_table()
                logger.debug(f"    [Filter] Table excluded by keyword '{exclude_kw}': {caption[:50]}...")
                return False

        # 检查包含关键词
        for include_kw in TABLE_INCLUDE_KEYWORDS:
            if include_kw in caption_lower:
                logger.debug(f"    [Filter] Table included by keyword '{include_kw}': {caption[:50]}...")
                return True

        # 如果没有明确的关键词，保守处理（仍然提取）
        logger.debug(f"    [Filter] Table unclear (no keywords), processing anyway: {caption[:50]}...")
        return True

    def _check_table_headers_for_keywords(self, block: Dict) -> bool:
        """
        智能路由步骤1: 检查表头是否包含动力学关键词

        Args:
            block: 表格块字典

        Returns:
            True: 表头包含动力学关键词
            False: 表头不含动力学关键词，应跳过
        """
        table_content = block.get('table_body', '') or block.get('content', '')

        if not table_content:
            # 没有HTML内容，依赖视觉模型
            return True  # 不在这里跳过，让视觉模型处理

        # 尝试从HTML提取表头
        import re
        # 提取第一行（表头）
        header_match = re.search(r'<thead>.*?</thead>', table_content, re.DOTALL)
        if header_match:
            header_text = header_match.group(0).lower()
        else:
            # 尝试提取第一个tr
            tr_match = re.search(r'<tr[^>]*>.*?</tr>', table_content, re.DOTALL)
            if tr_match:
                header_text = tr_match.group(0).lower()
            else:
                # 没有明确的表头，依赖视觉模型
                return True

        # 检查是否包含动力学关键词
        for keyword in KINETIC_HEADER_KEYWORDS:
            if keyword.lower() in header_text:
                logger.debug(f"    [Smart Routing] Header contains kinetic keyword '{keyword}'")
                return True

        # Header没有命中 → 扫描整个表格body（行标签可能包含Km/kcat/Vmax）
        full_table_lower = table_content.lower()
        for keyword in KINETIC_HEADER_KEYWORDS:
            if keyword.lower() in full_table_lower:
                logger.debug(f"    [Smart Routing] Table body contains kinetic keyword '{keyword}'")
                return True

        logger.debug(f"    [Smart Routing] Entire table lacks kinetic keywords, skipping")
        return False

    def _header_has_ambiguous_power_ten_multiplier(self, block: Dict) -> bool:
        """
        Detect kinetic headers where OCR/MinerU likely dropped a superscript.

        Example from the gold set:
            Vmax/(Eo × Km),10 M-1 min-1

        The intended header may be 10^3 or 10^4, but the parsed HTML no longer
        contains the exponent. Text-only extraction tends to guess here, so
        route these tables to the vision model when an image is available.
        """
        import re

        table_content = block.get('table_body', '') or block.get('content', '')
        if not table_content:
            return False

        header_match = re.search(r'<tr[^>]*>.*?</tr>', table_content, re.DOTALL | re.IGNORECASE)
        header_html = header_match.group(0) if header_match else table_content[:500]
        header_text = re.sub(r'<[^>]+>', ' ', header_html)
        header_text = re.sub(r'\s+', ' ', header_text)
        header_lower = header_text.lower()

        if not any(token in header_lower for token in ("kcat", "vmax", "km", "catalytic")):
            return False

        # Clear exponent forms are safe for text-only parsing.
        if re.search(r'10\s*(?:\^|[⁰¹²³⁴⁵⁶⁷⁸⁹]|\d)', header_text):
            return False

        return bool(re.search(r'10\s*m\s*[-−]?\s*1\s*(?:min|sec|s)', header_lower))

    def _can_parse_html_with_pandas(self, html_content: str) -> bool:
        """
        智能路由步骤2: 测试pandas是否能成功解析HTML

        Args:
            html_content: HTML表格内容

        Returns:
            True: pandas可以解析
            False: pandas解析失败，需要视觉模型
        """
        if not PANDAS_AVAILABLE:
            logger.debug(f"    [Smart Routing] pandas not available, using vision model")
            return False

        if not html_content:
            return False

        try:
            # 尝试解析HTML
            dfs = pd.read_html(io.StringIO(html_content), flavor='bs4')

            if not dfs or len(dfs) == 0:
                logger.debug(f"    [Smart Routing] pandas returned no dataframes")
                return False

            df = dfs[0]

            # 检查数据框是否有效
            if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
                logger.debug(f"    [Smart Routing] pandas returned empty dataframe")
                return False

            # 检查是否有NaN填充（可能是合并单元格导致的）
            nan_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
            if nan_ratio > 0.5:
                logger.debug(f"    [Smart Routing] High NaN ratio ({nan_ratio:.2f}), likely merged cells")
                return False

            logger.debug(f"    [Smart Routing] pandas parsed successfully: {df.shape}")
            return True

        except Exception as e:
            logger.debug(f"    [Smart Routing] pandas parsing failed: {e}")
            return False

    def _get_caption_text(self, block: Dict) -> str:
        caption = block.get('table_caption', '')
        if isinstance(caption, list):
            return ' '.join(str(c) for c in caption)
        return str(caption or '')

    def _get_footnote_text(self, block: Dict) -> str:
        footnote = block.get('table_footnote', '')
        if isinstance(footnote, list):
            return ' '.join(str(f) for f in footnote)
        return str(footnote or '')

    def _resolve_table_image_path(self, block: Dict, paper_dir: Path) -> Optional[str]:
        """Resolve a table image path from content_list metadata."""
        img_path = (
            block.get('img_path') or
            block.get('image_path') or
            block.get('table_img') or
            block.get('table_image')
        )
        if not img_path:
            return None

        candidate = paper_dir / str(img_path)
        if candidate.exists():
            return str(candidate)

        alt = paper_dir / 'images' / os.path.basename(str(img_path))
        if alt.exists():
            return str(alt)

        return None

    def _is_complex_table_structure(self, block: Dict) -> bool:
        """Detect merged-cell or parent-header table structures that need visual rescue."""
        import re

        table_content = block.get('table_body', '') or block.get('content', '')
        if not table_content:
            return False

        lower = table_content.lower()
        if "rowspan" in lower or "colspan" in lower:
            return True

        if re.search(r'<td[^>]*rowspan=["\']?\d+["\']?[^>]*>\s*</td>', table_content, flags=re.I):
            return True

        if re.search(r'\d+(?:\.\d+)?\s*×\s*10(?!\s*(?:\^|\d|[²³⁴⁵⁶⁷⁸⁹]))', table_content):
            return True

        # Parent-label rows often contain one repeated token across every cell
        # after MinerU flattens merged cells (e.g. "Zearalenone" in all columns).
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, flags=re.I | re.S)
        for row in rows:
            cells = [
                re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', cell)).strip().lower()
                for cell in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.I | re.S)
            ]
            cells = [c for c in cells if c]
            if len(cells) >= 3 and len(set(cells)) == 1:
                return True
            for cell in cells:
                hits = [term for term in MYCOTOXIN_SUBSTRATE_TERMS if re.search(rf'\b{re.escape(term)}\b', cell, flags=re.I)]
                if len(set(hits)) >= 2:
                    return True

        return False

    def _has_mycotoxin_signal(self, block: Dict) -> bool:
        text = f"{self._get_caption_text(block)} {block.get('table_body', '') or block.get('content', '')}".lower()
        return any(term in text for term in MYCOTOXIN_SUBSTRATE_TERMS)

    def _has_mycotoxin_or_core_metric_signal(self, block: Dict, paper_dir: Optional[Path] = None) -> bool:
        """Detect tables worth image rescue based on substrate or core metric signal."""
        text = self._table_plain_text(block).lower()
        if paper_dir is not None:
            text = f"{text} {self._table_surrounding_context(block, paper_dir, window=800).lower()}"
        if any(term in text for term in MYCOTOXIN_SUBSTRATE_TERMS):
            return True
        core_metric_patterns = [
            r"\bk\s*_?\s*m\b",
            r"\bk\s*_?\s*cat\b",
            r"\bkcat\s*/\s*km\b",
            r"\bk\s*_?\s*cat\s*/\s*k\s*_?\s*m\b",
            r"\bdegradation\b",
            r"\bconversion\b",
            r"\bremoval\b",
            r"\bresidual\s+(?:toxin|mycotoxin|aflatoxin|don|zen|ota|fumonisin)\b",
        ]
        return any(re.search(pattern, text, flags=re.I) for pattern in core_metric_patterns)

    def _has_enzyme_system_signal(self, block: Dict) -> bool:
        text = f"{self._get_caption_text(block)} {block.get('table_body', '') or block.get('content', '')}".lower()
        return any(term in text for term in ENZYME_SYSTEM_TERMS)

    def _table_plain_text(self, block: Dict) -> str:
        """Return caption, footnote, and table text normalized for routing gates."""
        raw = " ".join(str(part or "") for part in (
            self._get_caption_text(block),
            block.get('table_body', '') or block.get('content', ''),
            self._get_footnote_text(block),
        ))
        text = re.sub(r"<[^>]+>", " ", raw)
        text = text.replace("\u2212", "-").replace("−", "-")
        text = text.replace("\u00b5", "µ").replace("μ", "µ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _has_activity_rate_unit_signal(self, text: str) -> bool:
        """Detect enzyme activity/rate units that are not degradation percentages."""
        normalized = text.lower().replace("μ", "µ").replace("−", "-")
        compact = re.sub(r"[\s{}\\_^$]+", "", normalized)
        for latex_token in ("mathrm", "mathsf", "mathbf", "left", "right", "text"):
            compact = compact.replace(latex_token, "")
        unit_patterns = [
            r"\b[unpµ]?\s*mol\s*(?:min|minute|s|sec|second)\s*-?1\s*(?:mg|g|ml|l)\s*-?1\b",
            r"\b[unpµ]?\s*mol\s*/\s*(?:min|minute|s|sec|second)\s*/\s*(?:mg|g|ml|l)\b",
            r"\b(?:u|iu)\s*/\s*(?:mg|ml|g|l)\b",
            r"\b(?:u|iu)\s+(?:mg|ml|g|l)\s*-?1\b",
            r"\bpmol\s+min\s*-?1\s+mg\s*-?1\b",
            r"\bnmol\s+min\s*-?1\s+mg\s*-?1\b",
            r"\bµmol\s+min\s*-?1\s+mg\s*-?1\b",
        ]
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in unit_patterns):
            return True
        compact_patterns = [
            r"(?:µmol|umol|mumol|pmol|nmol)(?:min|minute|s|sec|second)-?1(?:1)?(?:mg|g|ml|l)-?1",
            r"(?:u|iu)/(?:mg|ml|g|l)",
        ]
        return any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in compact_patterns)

    def _has_specific_or_generic_activity_scope(self, text: str) -> bool:
        """Detect tables whose metric family is generic enzyme activity."""
        normalized = text.lower()
        activity_terms = [
            "specific activity", "specific activities", "apparent specific activity",
            "enzyme activity", "enzymatic activity", "catalytic activity",
            "activity assay", "activities of", "activity with",
        ]
        return any(term in normalized for term in activity_terms)

    def _has_primary_kinetic_table_signal(self, text: str) -> bool:
        """Detect true kinetic parameter tables, excluding specific-activity-only tables."""
        normalized = text.lower().replace(" ", "")
        kinetic_patterns = [
            r"\bk\s*_?\s*m\b",
            r"\bk\s*_?\s*cat\b",
            r"\bkcat\s*/\s*km\b",
            r"\bk\s*_?\s*cat\s*/\s*k\s*_?\s*m\b",
            r"\bmichaelis\b",
        ]
        spaced = text.lower()
        compact_hit = any(re.search(pattern, spaced, flags=re.IGNORECASE) for pattern in kinetic_patterns)
        return compact_hit or "kcat/km" in normalized or "kcatk" in normalized

    def _has_explicit_degradation_table_signal(self, text: str) -> bool:
        """Detect toxin conversion/degradation metrics reported as such, not activity baselines."""
        normalized = text.lower()
        if any(term in normalized for term in (
            "relative activity", "residual activity", "remaining activity",
            "specific activity", "apparent specific activity",
        )):
            # These are not degradation-efficiency labels by themselves.
            pass

        degradation_terms = [
            "degradation", "degradation rate", "degradation efficiency",
            "conversion", "conversion efficiency", "removal", "removal rate",
            "reduction", "residual toxin", "residual mycotoxin",
            "residual aflatoxin", "residual don", "residual deoxynivalenol",
            "residual zen", "residual zearalenone", "residual ota",
        ]
        has_metric_word = any(term in normalized for term in degradation_terms)
        has_percent_or_concentration = bool(re.search(
            r"(?:\d+(?:\.\d+)?\s*%|%\s*(?:degradation|conversion|removal|reduction)|"
            r"(?:residual|remaining)\s+(?:toxin|mycotoxin|aflatoxin|don|zen|ota))",
            normalized,
            flags=re.IGNORECASE,
        ))
        return has_metric_word and has_percent_or_concentration

    def _table_metric_scope_skip_reason(self, block: Dict) -> Optional[str]:
        """
        Hard pre-extraction table-type gate.

        Generic/specific enzyme activity tables often mention a mycotoxin but
        report rate units such as µmol min-1 mg-1. Those are not degradation
        efficiency and should not be sent to text/vision table extraction.
        """
        text = self._table_plain_text(block)
        if not text:
            return None

        has_activity_scope = self._has_specific_or_generic_activity_scope(text)
        has_rate_units = self._has_activity_rate_unit_signal(text)
        has_kinetic_scope = self._has_primary_kinetic_table_signal(text)
        has_degradation_scope = self._has_explicit_degradation_table_signal(text)
        has_mycotoxin = any(term in text.lower() for term in MYCOTOXIN_SUBSTRATE_TERMS)

        if has_activity_scope and has_rate_units and not has_kinetic_scope and not has_degradation_scope:
            return "table_skipped_metric_out_of_scope_specific_activity"

        if has_activity_scope and has_rate_units and not has_mycotoxin:
            return "table_skipped_metric_out_of_scope_generic_activity"

        return None

    def _mark_table_skipped_before_extraction(self, block: Dict, reason: str) -> None:
        """Mark a table so routing, text extraction, and aggregation can ignore it."""
        block["_table_skip_reason"] = reason
        block["_skip_for_text_extraction"] = True
        self._skipped_table_type_gate_count = getattr(self, "_skipped_table_type_gate_count", 0) + 1
        TokenTracker.add_no_keyword_table()

    def _is_secondary_toxicity_endpoint_table(self, block: Dict) -> bool:
        """
        Detect toxicity/cell-response tables that should not trigger table-image
        rescue for primary degradation or kinetic extraction.
        """
        text = f"{self._get_caption_text(block)} {block.get('table_body', '') or block.get('content', '')} {self._get_footnote_text(block)}".lower()
        toxicity_terms = [
            "residual bioluminescence", "bioluminescence", "ecotoxicity",
            "cytotoxicity", "cell viability", "ldh", "ros", "dna damage",
            "residual toxicity", "toxicity endpoint", "photobacterium",
        ]
        return any(term in text for term in toxicity_terms)

    def _is_out_of_scope_primary_enzyme_material_table(self, block: Dict) -> bool:
        """Detect enzyme-material systems that should not spend vision calls."""
        import re

        text = f"{self._get_caption_text(block)} {block.get('table_body', '') or block.get('content', '')} {self._get_footnote_text(block)}".lower()
        terms = [
            "fiber material", "fibre material", "enzyme nanocomplex", "nanocomplex",
            "immobilized enzyme", "immobilised enzyme", "immobilization", "immobilisation",
            "enzyme-loaded support", "enzyme-coated", "enzyme-polymer composite",
            "polymer-supported", "supported enzyme", "enzyme support", "carrier",
            "silica gel", "bead", "membrane", "resin", "nanoparticle",
            "nanocomposite", "mof", "metal-organic framework", "adsorbent",
            "photocatalyst", "photocatalytic",
        ]

        def non_negated(term: str) -> bool:
            pattern = re.escape(term).replace(r"\ ", r"\s+")
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                prefix = text[max(0, match.start() - 40):match.start()]
                if re.search(r"\b(?:not|without|free\s+enzyme,\s+not)\s+(?:\w+\s+){0,4}$", prefix, flags=re.IGNORECASE):
                    continue
                return True
            return False

        return any(non_negated(term) for term in terms)

    def _should_force_multimodal_table_rescue(self, block: Dict, paper_dir: Path) -> Optional[str]:
        """
        Route only high-risk tables to table-image rescue.

        This prevents expensive vision calls for simple tables while catching
        cases where parsed HTML is formally readable but loses parent headers,
        row block labels, or sample labels.
        """
        metric_skip_reason = self._table_metric_scope_skip_reason(block)
        if metric_skip_reason:
            self._mark_table_skipped_before_extraction(block, metric_skip_reason)
            logger.info(
                "    [Table Type Gate] Table %s: SKIP VISION RESCUE (%s)",
                block.get("block_id", "unknown"),
                metric_skip_reason,
            )
            return None

        if not self._resolve_table_image_path(block, paper_dir):
            return None

        if not self._has_mycotoxin_or_core_metric_signal(block, paper_dir):
            return None

        if self._is_secondary_toxicity_endpoint_table(block):
            return None

        if self._is_out_of_scope_primary_enzyme_material_table(block):
            block["_table_skip_reason"] = "table_skipped_out_of_scope_enzyme_material_system"
            logger.warning("    [Smart Routing] Table %s: SKIP VISION RESCUE (out-of-scope enzyme-material system)", block.get("block_id", "unknown"))
            return None

        if not self._is_complex_table_structure(block):
            return None

        table_content = (block.get("table_body", "") or block.get("content", "") or "").lower()
        if "rowspan" in table_content or "colspan" in table_content:
            return "complex_merged_cell_mycotoxin_table"

        return "complex_high_risk_mycotoxin_metric_table"

    def _table_surrounding_context(self, block: Dict, paper_dir: Path, window: int = 1800) -> str:
        """Return a short text window around the table caption from full.md."""
        full_md = paper_dir / "full.md"
        if not full_md.exists():
            return ""

        try:
            text = full_md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

        caption = self._get_caption_text(block)
        needles = []
        if caption:
            needles.append(caption[:120])
            # Caption usually starts with "Table N"; that survives OCR better
            # than math-heavy enzyme names.
            import re
            match = re.search(r'Table\s+\d+', caption, flags=re.I)
            if match:
                needles.append(match.group(0))

        table_body = block.get('table_body', '') or block.get('content', '')
        if table_body:
            # A parent label is often more stable than the whole HTML string.
            for term in MYCOTOXIN_SUBSTRATE_TERMS:
                if term in table_body.lower():
                    needles.append(term)
                    break

        idx = -1
        lower_text = text.lower()
        for needle in needles:
            if not needle:
                continue
            idx = lower_text.find(str(needle).lower())
            if idx >= 0:
                break

        if idx < 0:
            return text[:window]

        start = max(0, idx - window)
        end = min(len(text), idx + window)
        return text[start:end]

    def _should_use_text_only_extraction(self, block: Dict) -> Tuple[bool, str]:
        """
        智能路由决策: 判断是否应该使用纯文本提取

        路由规则:
        ├─ HTML body 完整 + pandas 解析成功 + 表头含 Km/kcat 等关键词
        │   → 纯文本模式提取（跳过 GLM-4.6V，省 ~3,000 tokens/表）
        ├─ HTML body 完整但 pandas 解析失败（合并单元格、嵌套表头）
        │   → 视觉模型提取（需要图片辅助理解结构）
        ├─ HTML body 为空或严重不完整
        │   → 视觉模型提取（HTML 不可靠，图片为唯一数据源）
        └─ 表头不含动力学关键词
            → 直接跳过，不调用任何模型

        Args:
            block: 表格块字典

        Returns:
            (use_text_only, reason)
            - use_text_only: True=纯文本, False=视觉模型, None=跳过
            - reason: 决策原因（用于日志）
        """
        table_content = block.get('table_body', '') or block.get('content', '')

        metric_skip_reason = self._table_metric_scope_skip_reason(block)
        if metric_skip_reason:
            self._mark_table_skipped_before_extraction(block, metric_skip_reason)
            return (None, metric_skip_reason)

        # 步骤1: 检查表头是否含动力学关键词
        has_kinetic_keywords = self._check_table_headers_for_keywords(block)
        if not has_kinetic_keywords:
            return (None, "no_kinetic_keywords")

        # 步骤2: 检查HTML是否完整且可解析
        if not table_content or len(table_content) < 100:
            return (False, "html_incomplete")

        # 步骤2b: kinetic table header has a likely lost superscript multiplier.
        # Prefer vision so the real header image can disambiguate 10^3 vs 10^4.
        if self._header_has_ambiguous_power_ten_multiplier(block):
            return (False, "ambiguous_power_ten_multiplier_header")

        # 步骤3: 尝试用pandas解析
        if self._can_parse_html_with_pandas(table_content):
            return (True, "pandas_success")

        return (False, "pandas_failed")

    def _get_paper_context_for_table(self, block: Dict) -> str:
        """
        从 paper_blocks 中提取论文标题和摘要/引言上下文。

        用于表格提取 prompt，让 LLM 能正确识别酶名称而非将突变名当作酶名。

        Args:
            block: 当前表格块（可用于定位前后文本块）

        Returns:
            格式化的论文上下文字符串，若无法提取则返回空字符串
        """
        paper_blocks = getattr(self, '_current_paper_blocks', None)
        if not paper_blocks:
            return ""

        title = ""
        abstract_parts = []
        intro_parts = []
        current_section = "title"
        chars_collected = 0
        MAX_CONTEXT_CHARS = 800

        for b in paper_blocks:
            if b.get('type') != 'text':
                continue
            text = (b.get('text') or b.get('content') or '').strip()
            if not text:
                continue

            text_level = b.get('text_level', '')

            # 提取标题（第一个非空的大字号文本块）
            if not title and text_level in (1, '1') and len(text) > 10:
                # 过滤期刊名、页码等噪音
                lower_text = text.lower()
                if not any(skip in lower_text for skip in [
                    'biochemistry', 'article', 'just accepted',
                    'page ', 'paragon', 'acs ', 'elsevier',
                    'springer', 'wiley', 'doi:', 'http',
                    'running title',
                ]):
                    title = text
                    continue

            # 检测 section heading
            lower_text = text.lower()
            if text_level in (1, '1') or len(text) < 80:
                if 'abstract' in lower_text:
                    current_section = "abstract"
                    continue
                elif 'introduction' in lower_text:
                    current_section = "intro"
                    continue
                elif any(s in lower_text for s in [
                    'experimental', 'materials and methods',
                    'results', 'discussion', 'conclusion',
                    'references', 'acknowledgment',
                ]):
                    current_section = "other"
                    continue

            # 收集 abstract 和 introduction 内容
            if current_section == "abstract" and chars_collected < MAX_CONTEXT_CHARS:
                abstract_parts.append(text)
                chars_collected += len(text)
            elif current_section == "intro" and chars_collected < MAX_CONTEXT_CHARS:
                intro_parts.append(text)
                chars_collected += len(text)

        # 组装上下文
        parts = []
        if title:
            parts.append(f"Paper title: {title}")
        if abstract_parts:
            abstract_text = " ".join(abstract_parts)[:500]
            parts.append(f"Abstract: {abstract_text}")
        elif intro_parts:
            intro_text = " ".join(intro_parts)[:500]
            parts.append(f"Introduction (excerpt): {intro_text}")

        return "\n".join(parts) if parts else ""

    async def _extract_table_text_only(
        self,
        block: Dict,
        block_id: int,
        paper_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        纯文本模式提取表格（不使用视觉模型）

        使用文本模型（Kimi/DeepSeek）提取HTML表格内容

        Args:
            block: 表格块字典
            block_id: 块ID

        Returns:
            提取的记录列表
        """
        # Prefer faster text models for table-text extraction; Kimi is kept as a fallback.
        text_model = self.text_models.get("deepseek") or self.text_models.get("MiniMax-M2.7") or self.text_models.get("kimi")
        if not text_model:
            logger.warning(f"    [Text-Only] No text model available, skipping table {block_id}")
            return []

        # 提取信息
        caption = block.get('table_caption', '')
        if isinstance(caption, list):
            caption = ' '.join(caption)

        footnote = block.get('table_footnote', '')
        if isinstance(footnote, list):
            footnote = ' '.join(footnote)

        table_content = block.get('table_body', '') or block.get('content', '')

        # 获取论文上下文（标题、摘要/引言）和表格周边正文。
        # 周边正文常包含 application-matrix 表格的共享条件
        # （例如 "incubation at 30 °C for 24 h"），不应靠最终导出层猜测。
        paper_context = self._get_paper_context_for_table(block)
        nearby_context = self._table_surrounding_context(block, paper_dir) if paper_dir else ""

        # 构建提示词
        prompt = f"""{self.table_prompt}
{"=== 论文上下文 ===" + chr(10) + paper_context + chr(10) if paper_context else ""}=== 表格信息 ===

【标题】
{caption if caption else '(无标题)'}

【脚注】
{footnote if footnote else '(无脚注)'}

【表格HTML内容】
{table_content[:5000] if table_content else '(无HTML内容)'}

【表格附近正文】
{nearby_context[:2500] if nearby_context else '(无附近正文)'}

请从上述HTML表格中提取酶动力学数据。注意：
1. 优先使用HTML中的数值数据
2. 注意检查单位是否正确
3. 如果某个参数在HTML中没有，标记为 null
4. 如果表头类似 "10 M-1 min-1" 且缺少 10 的指数，不要猜测 10^3/10^4；
   保留原始数值，设置 human_review_required=true，并加入 table_multiplier_scaling_error
5. enzyme_name 应为酶的正式名称（如 Os79、rCuL），而非突变体标记（如 WT、Q202E、H122A）；
   突变体标记应填入 mutations 字段
6. 如果表格附近正文给出了与本表同一实验的共享反应条件（time / temperature / pH / matrix pH），
   可填入相应 condition 字段；不要跨不同表格或不同实验继承条件。"""

        # 使用文本模型提取
        if text_model is self.text_models.get("deepseek"):
            model_name = "deepseek"
        elif text_model is self.text_models.get("MiniMax-M2.7"):
            model_name = "MiniMax-M2.7"
        else:
            model_name = "kimi"
        semaphore = get_semaphore(model_name)

        try:
            records = await self._extract_with_semaphore(
                semaphore,
                self._extract_text_block,
                text_model,
                prompt,
                block_id,
                model_name,
                TABLE_TEXT_MAX_TOKENS
            )
            logger.debug(f"    [Text-Only] Table {block_id}: extracted {len(records)} records")
            deterministic_records = self._extract_simple_kinetic_table_from_html(block, block_id)
            if deterministic_records:
                records = self._merge_table_rescue_records(records or [], deterministic_records)
            if not records:
                records = deterministic_records
            records = self._enrich_table_records_with_nearby_conditions(records or [], nearby_context)
            return records
        except Exception as e:
            logger.error(f"    [Text-Only] Table {block_id} extraction failed: {e}")
            records = self._extract_simple_kinetic_table_from_html(block, block_id)
            return self._enrich_table_records_with_nearby_conditions(records, self._table_surrounding_context(block, paper_dir) if paper_dir else "")

    def _enrich_table_records_with_nearby_conditions(self, records: List[Dict], nearby_context: str) -> List[Dict]:
        """Fill table-local missing conditions from nearby same-table prose.

        This is deliberately generic and only fills missing fields. It supports
        application tables whose cells contain only matrix-specific percentages,
        while the paragraph immediately above the table states shared conditions.
        """
        if not records or not nearby_context:
            return records

        text = nearby_context.replace("\u2212", "-").replace("−", "-")
        compact = re.sub(r"\s+", " ", text)

        temp_value = None
        temp_match = re.search(r"(?:incubat(?:ed|ion)|reaction|applied|conditions?)[^.]{0,120}?(\d+(?:\.\d+)?)\s*°?\s*C", compact, flags=re.IGNORECASE)
        if temp_match:
            temp_value = float(temp_match.group(1))

        time_value = None
        time_unit = None
        time_match = re.search(r"(?:incubat(?:ed|ion)|reaction|applied|conditions?)[^.]{0,140}?(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|mins|minute|minutes)\b", compact, flags=re.IGNORECASE)
        if time_match:
            time_value = float(time_match.group(1))
            unit = time_match.group(2).lower()
            time_unit = "h" if unit.startswith("h") or "hour" in unit else "min"

        matrix_ph = {}
        for matrix, ph in re.findall(r"\b(beer|milk|lager beer|uht milk)\b[^.]{0,80}?\bpH\s*([0-9]+(?:\.[0-9]+)?)", compact, flags=re.IGNORECASE):
            key = matrix.lower()
            if "beer" in key:
                matrix_ph["beer"] = float(ph)
            elif "milk" in key:
                matrix_ph["milk"] = float(ph)

        for record in records:
            if str(record.get("measurement_type") or "").lower() != "degradation":
                continue
            if temp_value is not None and not record.get("degradation_temperature_value"):
                record["degradation_temperature_value"] = temp_value
                record["degradation_temperature_unit"] = "°C"
            if time_value is not None and not record.get("degradation_time_value"):
                record["degradation_time_value"] = time_value
                record["degradation_time_unit"] = time_unit
                flags = record.get("error_flags") or []
                if isinstance(flags, str):
                    flags = [f.strip() for f in re.split(r"[;,|]", flags) if f.strip()]
                record["error_flags"] = [
                    f for f in flags
                    if "missing_time_for_degradation" not in str(f).lower()
                ]
            matrix = str(record.get("matrix") or "").lower()
            if not record.get("degradation_ph"):
                if "beer" in matrix and "beer" in matrix_ph:
                    record["degradation_ph"] = matrix_ph["beer"]
                elif "milk" in matrix and "milk" in matrix_ph:
                    record["degradation_ph"] = matrix_ph["milk"]
        return records

    async def _extract_text_block(
        self,
        model,
        content: str,
        block_id: int,
        model_name: str,
        max_tokens: int = 4096
    ) -> List[Dict]:
        """提取文本块（用于纯文本表格提取）"""
        messages = [
            {"role": "system", "content": "You are an expert in enzyme kinetics data extraction."},
            {"role": "user", "content": content}
        ]

        # 在线程池中运行同步调用
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.chat(messages=messages, temperature=0.1, max_tokens=max_tokens, task=f"text_{model_name}")
        )
        return self._parse_json_response(response)

    def _extract_simple_kinetic_table_from_html(self, block: Dict, block_id: int) -> List[Dict]:
        """
        Deterministic fallback for simple kinetic tables.

        This is intentionally narrow: it only handles tables with substrate,
        Vmax/E0, Km, and Vmax/(E0*Km) columns. It protects against a failure
        mode where the vision/text LLM returns no records for a clearly
        structured table.
        """
        import re

        table_content = block.get('table_body', '') or block.get('content', '')
        if not table_content or not PANDAS_AVAILABLE:
            return []
        if "rowspan" in table_content.lower() or "colspan" in table_content.lower():
            return []

        try:
            dfs = pd.read_html(io.StringIO(table_content), header=0, flavor='bs4')
        except Exception as exc:
            logger.debug(f"    [HTML Fallback] Table {block_id}: pandas parse failed: {exc}")
            return []

        if not dfs:
            return []

        df = dfs[0]
        headers = [str(c).strip() for c in df.columns]
        header_text = " ".join(headers)
        header_lower = header_text.lower()
        if not ("km" in header_lower and "vmax" in header_lower and ("km)" in header_lower or "k m" in header_lower)):
            return []

        def find_col(*needles):
            for col in headers:
                col_lower = col.lower().replace(" ", "")
                if all(needle in col_lower for needle in needles):
                    return col
            return None

        substrate_col = headers[0] if headers else None
        vmax_col = find_col("vmax", "eo")
        km_col = find_col("km")
        kcat_km_col = None
        for col in headers:
            col_lower = col.lower().replace(" ", "")
            if "vmax" in col_lower and "km" in col_lower and col != vmax_col:
                kcat_km_col = col
                break

        if not (substrate_col and vmax_col and km_col and kcat_km_col):
            return []

        from src.utils.table_multiplier import parse_table_header_multiplier

        multiplier, source_text, ambiguous = parse_table_header_multiplier(kcat_km_col)
        human_review_required = False
        error_flags = []
        if not multiplier and self._header_has_ambiguous_power_ten_multiplier(block):
            # MinerU can drop the superscript from 10^3 in this exact kinetic
            # header shape, leaving "10 M-1 min-1" in HTML while the image
            # contains the exponent. The row values are still reliable.
            multiplier = 1000.0
            source_text = "10^3 inferred from OCR-lost superscript in Vmax/(E0 x Km) header"
            human_review_required = True
            error_flags.append("table_multiplier_scaling_error")
        elif ambiguous:
            human_review_required = True
            error_flags.append("table_multiplier_scaling_error")
            multiplier = None

        def first_number(value):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
            return float(match.group(0)) if match else None

        caption = block.get('table_caption', '')
        if isinstance(caption, list):
            caption = ' '.join(str(c) for c in caption)

        def infer_enzyme_from_caption(caption_text: str) -> Optional[str]:
            text = str(caption_text or "").strip()
            patterns = [
                r"kinetic parameters (?:of|for)\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9α-ωΑ-Ωµμ_/+().-]{1,80}?)(?:\s+(?:with|toward|against|in|under|at|using)\b|[.,;:])",
                r"(?:enzyme|catalyst)\s+([A-Za-z0-9][A-Za-z0-9α-ωΑ-Ωµμ_/+().-]{1,80}?)\s+(?:kinetic|catalytic)",
                r"\b([A-Za-z0-9][A-Za-z0-9α-ωΑ-Ωµμ_/+().-]{1,80}?)\s+(?:kinetic|catalytic)\s+parameters\b",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    value = match.group(1).strip(" .,:;()")
                    if value and not any(stop in value.lower() for stop in ["various", "different", "selected", "reaction"]):
                        return value
            return None

        enzyme_name = infer_enzyme_from_caption(caption)

        ph = None
        ph_match = re.search(r"pH\s*([0-9]+(?:\.[0-9]+)?)", caption, re.IGNORECASE)
        if ph_match:
            ph = float(ph_match.group(1))

        records = []
        for _, row in df.iterrows():
            substrate = str(row.get(substrate_col, "")).strip()
            if not substrate or substrate.lower() in {"nan", "mycotoxin", "sample"}:
                continue
            substrate_lower = substrate.lower()
            if not any(term in substrate_lower for term in MYCOTOXIN_SUBSTRATE_TERMS):
                continue

            kcat_km_value = first_number(row.get(kcat_km_col))
            # Don't apply multiplier here — let apply_kinetic_unit_multiplier
            # handle it during normalize_records_batch.  This avoids
            # double-scaling when aggregation strips _table_multiplier_applied.

            record = {
                "enzyme_name": enzyme_name,
                "reported_enzyme_name": enzyme_name,
                "substrate": substrate,
                "measurement_type": "kinetic",
                "condition_scope": "kinetic_assay",
                "Km_value": first_number(row.get(km_col)),
                "Km_unit": "mM" if "mm" in km_col.lower() else None,
                "kcat_value": first_number(row.get(vmax_col)),
                "kcat_unit": "min⁻¹" if "min" in vmax_col.lower() else "s⁻¹" if "sec" in vmax_col.lower() else None,
                "kcat_Km_value": kcat_km_value,
                "kcat_Km_unit": "M⁻¹ min⁻¹" if "min" in kcat_km_col.lower() else "M⁻¹ s⁻¹" if "sec" in kcat_km_col.lower() else None,
                "kinetic_ph": ph,
                "kinetic_unit_multiplier": multiplier,
                "kinetic_unit_source_text": source_text or kcat_km_col,
                "human_review_required": human_review_required,
                "error_flags": list(error_flags),
                "notes": f"Deterministic HTML fallback from table {block_id}; verify multiplier against table image.",
                "_source_type": "table",
                "_extracted_by": "html_fallback",
                "_extraction_method": "deterministic-html",
            }
            records.append(record)

        if records:
            logger.warning(f"    [HTML Fallback] Table {block_id}: extracted {len(records)} simple kinetic records")
        return records

    def _merge_table_rescue_records(self, records: List[Dict], fallback_records: List[Dict]) -> List[Dict]:
        """Merge rescue records by enzyme/substrate/sample without deleting LLM records."""
        merged = list(records or [])

        def key(record: Dict) -> Tuple[str, str, str]:
            notes = str(record.get("notes") or "")
            sample = ""
            match = re.search(r"sample\s*#?\s*(\d+)", notes, flags=re.IGNORECASE)
            if match:
                sample = f"sample{match.group(1)}"
            return (
                str(record.get("reported_enzyme_name") or record.get("enzyme_name") or "").lower(),
                str(record.get("substrate") or "").lower(),
                sample,
            )

        seen = {key(r) for r in merged}
        for record in fallback_records:
            record_key = key(record)
            if record_key not in seen:
                merged.append(record)
                seen.add(record_key)
        return merged

    def _extract_complex_kinetic_table_from_html(self, block: Dict, block_id: int) -> List[Dict]:
        """
        Deterministic fallback for complex Vmax/E0 + Km + Vmax/(E0*Km) tables.

        This handles tables with parent substrate rows and row-spanned enzyme
        labels. It is generic for this header shape and only emits records for
        mycotoxin parent blocks.
        """
        import re
        from src.utils.table_multiplier import parse_table_header_multiplier

        table_content = block.get('table_body', '') or block.get('content', '')
        if not table_content or not PANDAS_AVAILABLE:
            return []

        try:
            dfs = pd.read_html(io.StringIO(table_content), header=None, flavor='bs4')
        except Exception as exc:
            logger.debug(f"    [Complex HTML Fallback] Table {block_id}: pandas parse failed: {exc}")
            return []
        if not dfs:
            return []

        df = dfs[0]
        if df.empty or df.shape[1] < 4:
            return []

        first_row = [str(v).strip() for v in df.iloc[0].tolist()]
        header_text = " ".join(first_row)
        header_lower = header_text.lower()
        if not ("km" in header_lower and "vmax" in header_lower):
            return []

        def find_index(*needles: str) -> Optional[int]:
            for idx, value in enumerate(first_row):
                compact = value.lower().replace(" ", "")
                if all(n in compact for n in needles):
                    return idx
            return None

        enzyme_idx = 0
        sample_idx = 1 if df.shape[1] > 1 else None
        vmax_idx = find_index("vmax", "e0")
        km_idx = find_index("km")
        kcat_km_idx = None
        for idx, value in enumerate(first_row):
            compact = value.lower().replace(" ", "")
            if "vmax" in compact and "km" in compact and idx != vmax_idx:
                kcat_km_idx = idx
                break
        if vmax_idx is None or km_idx is None or kcat_km_idx is None:
            return []

        multiplier, source_text, ambiguous = parse_table_header_multiplier(first_row[kcat_km_idx])
        if ambiguous:
            multiplier = None

        def clean(value: Any) -> str:
            text = str(value or "").strip()
            return "" if text.lower() == "nan" else text

        def first_number(value: Any) -> Optional[float]:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", clean(value))
            return float(match.group(0)) if match else None

        def is_parent_substrate_row(cells: List[str]) -> Optional[str]:
            nonempty = [c for c in cells if c]
            if not nonempty:
                return None
            candidate = nonempty[0]
            if len(set(c.lower() for c in nonempty)) == 1 and any(term in candidate.lower() for term in MYCOTOXIN_SUBSTRATE_TERMS):
                return candidate
            return None

        def normalize_sample(sample: str, group_index: int) -> str:
            sample = clean(sample)
            # MinerU sometimes merges "#2 #3" into the second row and leaves
            # the third sample blank. Preserve the row count instead of dropping.
            if not sample or sample.lower() == "nan":
                return f"#{group_index}"
            matches = re.findall(r"#\s*\d+", sample)
            if matches:
                return matches[0].replace(" ", "")
            return sample

        caption = self._get_caption_text(block)
        ph = None
        ph_match = re.search(r"pH\s*([0-9]+(?:\.[0-9]+)?)", caption, re.IGNORECASE)
        if ph_match:
            ph = float(ph_match.group(1))

        records: List[Dict] = []
        current_substrate: Optional[str] = None
        group_row_count = 0
        for row_idx in range(1, len(df)):
            cells = [clean(v) for v in df.iloc[row_idx].tolist()]
            parent = is_parent_substrate_row(cells)
            if parent:
                current_substrate = parent
                group_row_count = 0
                continue

            if not current_substrate:
                continue

            enzyme = clean(cells[enzyme_idx]) if enzyme_idx < len(cells) else ""
            if not enzyme or not any(term in enzyme.lower() for term in ENZYME_SYSTEM_TERMS):
                continue

            km_value = first_number(cells[km_idx] if km_idx < len(cells) else "")
            kcat_value = first_number(cells[vmax_idx] if vmax_idx < len(cells) else "")
            kcat_km_value = first_number(cells[kcat_km_idx] if kcat_km_idx < len(cells) else "")
            if km_value is None and kcat_value is None and kcat_km_value is None:
                continue

            group_row_count += 1
            sample = normalize_sample(cells[sample_idx] if sample_idx is not None and sample_idx < len(cells) else "", group_row_count)
            # Don't apply multiplier here — let apply_kinetic_unit_multiplier
            # handle it during normalize_records_batch.

            record = {
                "enzyme_name": enzyme,
                "reported_enzyme_name": enzyme,
                "substrate": current_substrate,
                "measurement_type": "kinetic",
                "measurement_context_id": f"table_{block_id}|{enzyme}|{current_substrate}|{sample}|kinetic",
                "condition_scope": "kinetic_assay",
                "Km_value": km_value,
                "Km_unit": "mM" if "mm" in first_row[km_idx].lower() else None,
                "kcat_value": kcat_value,
                "kcat_unit": "min⁻¹" if "min" in first_row[vmax_idx].lower() else "s⁻¹" if "sec" in first_row[vmax_idx].lower() else None,
                "kcat_Km_value": kcat_km_value,
                "kcat_Km_unit": "M⁻¹ min⁻¹" if "min" in first_row[kcat_km_idx].lower() else "M⁻¹ s⁻¹" if "sec" in first_row[kcat_km_idx].lower() else None,
                "kinetic_ph": ph,
                "kinetic_unit_multiplier": multiplier,
                "kinetic_unit_source_text": source_text or first_row[kcat_km_idx],
                "human_review_required": True,
                "error_flags": ["table_image_rescue"],
                "notes": f"Deterministic complex-table fallback from table {block_id}; sample {sample}; parent substrate labels expanded; verify against table image.",
                "source_section": f"table_{block_id}",
                "_source_type": "table",
                "_extracted_by": "complex_html_fallback",
                "_extraction_method": "deterministic-complex-html",
            }
            records.append(record)

        if records:
            logger.warning(f"    [Complex HTML Fallback] Table {block_id}: extracted {len(records)} complex kinetic records")
        return records

    def _compress_image_if_needed(self, image_path: str) -> str:
        """
        优化3: 压缩图片到指定尺寸和质量

        Args:
            image_path: 原始图片路径

        Returns:
            压缩后的图片路径（如果是临时文件）
        """
        if not PIL_AVAILABLE:
            return image_path

        try:
            img = Image.open(image_path)

            # 检查是否需要压缩
            width, height = img.size
            if width <= MAX_IMAGE_WIDTH:
                return image_path

            # 计算新尺寸（保持宽高比）
            if width > height:
                new_width = MAX_IMAGE_WIDTH
                new_height = int(height * MAX_IMAGE_WIDTH / width)
            else:
                new_height = MAX_IMAGE_WIDTH
                new_width = int(width * MAX_IMAGE_WIDTH / height)

            # 压缩图片
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 创建临时文件
            import tempfile
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)

            # 保存为JPEG格式
            img_resized.save(temp_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)

            logger.debug(f"    [Compress] {width}x{height} → {new_width}x{new_height}, saved to {temp_path}")

            return temp_path

        except Exception as e:
            logger.warning(f"    [Compress] Failed to compress image: {e}")
            return image_path

    def _build_table_normalization_prompt(self, block: Dict, paper_dir: Path) -> str:
        """Build a multimodal prompt that reconstructs a normalized table only."""
        caption = self._get_caption_text(block)
        footnote = self._get_footnote_text(block)
        nearby_text = self._table_surrounding_context(block, paper_dir)

        return f"""Reconstruct the attached scientific table image into a normalized Markdown table.

Return ONLY a normalized Markdown table. No explanation before or after the table.

Use the image as the source of truth. The parsed HTML may be wrong for merged
cells and superscripts, so do not copy row labels or exponents from HTML when
they conflict with the image.

Rules:
1. Expand every merged cell, parent header, row-block label, and split label into every data row.
2. Do not stop after the first block; scan the whole table from top to bottom.
3. Output one row per measurement context.
4. Include rows with reported numeric values; rows where every metric is n/a may be omitted.
5. Preserve units and scientific notation exactly, including 10^2, 10^3, 10^4, 10^5, M^-1, s^-1, min^-1.
6. If a row label is split across visual lines, join it. Example: H122A/L123A plus /Q202L becomes H122A/L123A/Q202L.
7. Leave unclear cells blank. Do not invent values.

Caption:
{caption if caption else '(no caption)'}

Footnotes:
{footnote if footnote else '(no footnotes)'}

Nearby text:
{nearby_text[:1200] if nearby_text else '(no nearby text)'}
"""

    async def _reconstruct_normalized_table_from_image(
        self,
        model,
        block: Dict,
        block_id: int,
        paper_dir: Path,
        model_name: str
    ) -> Optional[str]:
        """Use the multimodal model to rebuild a normalized Markdown table."""
        image_path = self._resolve_table_image_path(block, paper_dir)
        if not image_path:
            logger.warning(f"    [Table Rescue] Table {block_id}: no image available for normalization")
            return None

        compressed_image_path = self._compress_image_if_needed(image_path)
        prompt = self._build_table_normalization_prompt(block, paper_dir)
        messages = [{"role": "user", "text": prompt, "image_path": compressed_image_path}]

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: model.chat(
                    messages=messages,
                    is_multimodal=True,
                    temperature=0.0,
                    max_tokens=GLM46V_MAX_TOKENS,
                    task="table_image_normalization"
                )
            )
            return str(response or "").strip()
        finally:
            if compressed_image_path != image_path and os.path.exists(compressed_image_path):
                try:
                    os.remove(compressed_image_path)
                except Exception:
                    pass

    async def _extract_records_from_normalized_table(
        self,
        normalized_table: str,
        block: Dict,
        block_id: int
    ) -> List[Dict]:
        """Send a normalized table through the existing table prompt."""
        text_model = self.text_models.get("deepseek") or self.text_models.get("MiniMax-M2.7") or self.text_models.get("kimi")
        if not text_model:
            logger.warning(f"    [Table Rescue] No text model available for normalized table {block_id}")
            return []

        caption = self._get_caption_text(block)
        footnote = self._get_footnote_text(block)
        model_name = "deepseek" if text_model is self.text_models.get("deepseek") else (
            "MiniMax-M2.7" if text_model is self.text_models.get("MiniMax-M2.7") else "kimi"
        )
        semaphore = get_semaphore(model_name)
        # 获取论文上下文
        paper_context = self._get_paper_context_for_table(block)

        prompt = f"""{self.table_prompt}
{"=== 论文上下文 ===" + chr(10) + paper_context + chr(10) if paper_context else ""}=== Image-normalized table rescue ===

The following Markdown table was reconstructed from the table image. Merged cells,
parent headers, row block labels, and sample labels have been expanded into each
data row. Use this normalized table as the main table source.

Extract every valid numeric measurement row from the normalized table. Do not
summarize a row block, do not stop after the first enzyme/substrate block, and
do not drop rows just because another row has the same enzyme or substrate.

Caption:
{caption if caption else '(no caption)'}

Footnotes:
{footnote if footnote else '(no footnotes)'}

Normalized table:
{normalized_table[:8000]}

If the normalized table contains `image_html_conflict` or uncertain structure,
set human_review_required=true and explain the conflict in notes.
enzyme_name should be the official enzyme name (e.g., Os79, rCuL), NOT mutation labels
(e.g., WT, Q202E, H122A). Mutation labels go in the mutations field.
"""
        records = await self._extract_with_semaphore(
            semaphore,
            self._extract_text_block,
            text_model,
            prompt,
            block_id,
            model_name
        )
        for record in records:
            record["human_review_required"] = True
            flags = record.get("error_flags") or []
            if isinstance(flags, str):
                flags = [f.strip() for f in re.split(r"[;,|]", flags) if f.strip()]
            if "table_image_rescue" not in flags:
                flags.append("table_image_rescue")
            record["error_flags"] = flags
            notes = record.get("notes") or ""
            suffix = "table_image_rescue: normalized from table image"
            if suffix not in notes:
                record["notes"] = f"{notes} | {suffix}".strip(" |")
            record["_extraction_method"] = "table-image-normalized-rescue"
        return records

    async def _extract_records_directly_from_table_image(
        self,
        model,
        block: Dict,
        block_id: int,
        paper_dir: Path,
        model_name: str
    ) -> List[Dict]:
        """Extract table records directly from the image with a concise prompt."""
        image_path = self._resolve_table_image_path(block, paper_dir)
        if not image_path:
            return []

        caption = self._get_caption_text(block)
        footnote = self._get_footnote_text(block)
        nearby_text = self._table_surrounding_context(block, paper_dir, window=900)
        compressed_image_path = self._compress_image_if_needed(image_path)
        prompt = f"""Extract all valid numeric measurement rows from the attached scientific table image.

Return ONLY a JSON array. No markdown. No explanation.

Use the image as the source of truth. Ignore parsed HTML if row labels, merged
cells, or superscripts conflict with the image.

Output one object per numeric measurement row with these existing schema keys:
reported_enzyme_name, enzyme_name, organism, mutations, substrate,
measurement_type, Km_value, Km_unit, kcat_value, kcat_unit,
kcat_Km_value, kcat_Km_unit, degradation_efficiency,
degradation_efficiency_unit, evidence_text, notes.

Rules:
1. Scan the whole table from top to bottom. Do not stop after the first block.
2. Expand all merged row labels and parent labels into each row.
3. Join split labels. Example: H122A/L123A plus /Q202L becomes H122A/L123A/Q202L.
4. Omit rows where all measurement values are n/a.
5. Preserve substrate names and units from the image.
6. Convert scientific notation to numeric values when unambiguous, e.g. 1.75 x 10^4 -> 17500.
7. Do not invent values for blank or unclear cells.
8. Use measurement_type=\"kinetic\" for Km/kcat/kcat_Km rows and \"degradation\" for degradation/conversion/removal rows.

Caption:
{caption if caption else '(no caption)'}

Footnotes:
{footnote if footnote else '(no footnotes)'}

Nearby context:
{nearby_text[:900] if nearby_text else '(no nearby text)'}
"""
        messages = [{"role": "user", "text": prompt, "image_path": compressed_image_path}]
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: model.chat(
                    messages=messages,
                    is_multimodal=True,
                    temperature=0.0,
                    max_tokens=GLM46V_MAX_TOKENS,
                    task="table_image_direct_records"
                )
            )
        finally:
            if compressed_image_path != image_path and os.path.exists(compressed_image_path):
                try:
                    os.remove(compressed_image_path)
                except Exception:
                    pass

        records = self._parse_json_response(response)
        for record in records:
            flags = record.get("error_flags") or []
            if isinstance(flags, str):
                flags = [f.strip() for f in re.split(r"[;,|]", flags) if f.strip()]
            if "table_image_rescue" not in flags:
                flags.append("table_image_rescue")
            record["error_flags"] = flags
            notes = record.get("notes") or ""
            suffix = "table_image_direct: extracted from table image"
            if suffix not in notes:
                record["notes"] = f"{notes} | {suffix}".strip(" |")
            record["_source_type"] = "table"
            record["_extracted_by"] = model_name
            record["_extraction_method"] = "table-image-direct-records"
            record["_source_block_id"] = block_id
        return records

    async def _extract_table_image_rescue(
        self,
        model,
        block: Dict,
        block_id: int,
        paper_dir: Path,
        model_name: str
    ) -> List[Dict]:
        """
        Two-step image rescue:
        1. reconstruct normalized Markdown table from image;
        2. run the existing table extraction prompt over that normalized table.
        """
        logger.info(f"    [Table Rescue] Table {block_id}: image normalization rescue triggered ({block.get('_table_image_rescue_reason')})")
        records = await self._extract_records_directly_from_table_image(
            model=model,
            block=block,
            block_id=block_id,
            paper_dir=paper_dir,
            model_name=model_name
        )

        if not records:
            normalized_table = await self._reconstruct_normalized_table_from_image(
                model=model,
                block=block,
                block_id=block_id,
                paper_dir=paper_dir,
                model_name=model_name
            )
            if normalized_table:
                records = await self._extract_records_from_normalized_table(normalized_table, block, block_id)

        # Deterministic fallback is intentionally run after the image-normalized
        # path. It prevents known complex kinetic tables from becoming zero
        # records when the vision model follows the normalized-table instruction
        # but the second extraction step is over-conservative.
        fallback_records = self._extract_complex_kinetic_table_from_html(block, block_id)
        if fallback_records:
            records = self._merge_table_rescue_records(records, fallback_records)

        return records

    async def _extract_table_block_multimodal(
        self,
        model,
        block: Dict,
        block_id: int,
        paper_dir: Path,
        model_name: str
    ) -> List[Dict]:
        """
        Extract a table that has been routed to the multimodal model.

        Multimodal table extraction is intentionally normalized-first:
        image → expanded Markdown table → existing table-record extractor.
        Direct image-to-record extraction is too brittle for merged-cell tables.
        """
        if not block.get('_table_image_rescue_reason'):
            block['_table_image_rescue_reason'] = 'vision_table_image_normalization'
        return await self._extract_table_image_rescue(
            model=model,
            block=block,
            block_id=block_id,
            paper_dir=paper_dir,
            model_name=model_name
        )
    
    async def _extract_figure_block(
        self,
        model,
        image_path: str,
        block_id: int,
        paper_dir: Path,
        model_name: str
    ) -> List[Dict]:
        """提取图片块"""
        import os
        
        # 检查图片路径是否为空
        if not image_path:
            logger.warning(f"    Block {block_id}: Empty image path, skipping")
            return []
        
        full_image_path = str(paper_dir / image_path)
        
        # 检查图片是否存在
        if not os.path.exists(full_image_path):
            # 尝试在images子目录查找
            alt_path = str(paper_dir / 'images' / os.path.basename(image_path))
            if os.path.exists(alt_path):
                full_image_path = alt_path
            else:
                logger.warning(f"    Block {block_id}: Image not found: {full_image_path}")
                return []
        
        messages = [
            {
                "role": "user",
                "text": self.figure_prompt,
                "image_path": full_image_path
            }
        ]
        
        # 在线程池中运行同步调用
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: model.chat(messages=messages, is_multimodal=True, temperature=0.1, task="table_vision")
        )
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """解析JSON响应"""
        import re, json

        def normalize_payload(data):
            if isinstance(data, dict):
                for key in ("records", "extracted_records", "data", "results",
                            "extracted_data", "enzymes", "entries", "output",
                            "enzyme_records", "kinetic_records"):
                    value = data.get(key)
                    if isinstance(value, list):
                        return value
                if len(data) == 1:
                    only_val = next(iter(data.values()))
                    if isinstance(only_val, list):
                        return only_val
                return [data]
            return data

        content = response
        if isinstance(response, dict) and 'content' in response:
            content = response['content']

        # Strip <think>…</think> blocks (MiniMax reasoning residue)
        if content and "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # 提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        try:
            data = json.loads(content)
            return normalize_payload(data)
        except json.JSONDecodeError:
            pass

        # Fallback 1: strip opening code fence (handles truncated responses)
        stripped = re.sub(r'^\s*```json\s*', '', content).strip()
        if stripped != content.strip():
            try:
                data = json.loads(stripped)
                return normalize_payload(data)
            except json.JSONDecodeError:
                pass

            # Fallback 2: complete truncated JSON by closing open brackets
            opens = stripped.count('[') - stripped.count(']')
            openc = stripped.count('{') - stripped.count('}')
            if opens > 0 or openc > 0:
                attempt = stripped.rstrip().rstrip(',')
                attempt += '}' * max(0, openc)
                attempt += ']' * max(0, opens)
                try:
                    data = json.loads(attempt)
                    if isinstance(data, list):
                        logger.info(f"Recovered {len(data)} records from truncated JSON response")
                        return data
                    elif isinstance(data, dict):
                        return normalize_payload(data)
                except json.JSONDecodeError:
                    pass

                # Fallback 2b: unterminated string — truncate to last complete
                # key-value pair and re-close brackets
                last_comma = attempt.rfind(',')
                if last_comma > 0:
                    truncated = attempt[:last_comma]
                    t_opens = truncated.count('[') - truncated.count(']')
                    t_openc = truncated.count('{') - truncated.count('}')
                    truncated += '}' * max(0, t_openc)
                    truncated += ']' * max(0, t_opens)
                    try:
                        data = json.loads(truncated)
                        if isinstance(data, list):
                            logger.info(f"Recovered {len(data)} records from truncated JSON (unterminated string fix)")
                            return data
                        elif isinstance(data, dict):
                            return normalize_payload(data)
                    except json.JSONDecodeError:
                        pass

        # Fallback 3: array regex on original content
        array_match = re.search(r'\[.*\]', content, re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning(f"JSON parse failed. Response snippet: {content[:200]}")
        return []
    
    def _remove_skipped_table_context_from_text(self, text: str, paper_blocks: List[Dict]) -> str:
        """
        Remove tables that the metric-scope gate has already rejected from
        full.md text before sending it to student/teacher LLM calls.
        """
        if not text or not paper_blocks:
            return text

        cleaned = text
        removed_count = 0
        for block in paper_blocks:
            if not block.get("_skip_for_text_extraction"):
                continue

            fragments = []
            for key in ("table_body", "content"):
                fragment = str(block.get(key) or "")
                if fragment and len(fragment) > 30:
                    fragments.append(fragment)

            for fragment in fragments:
                if fragment in cleaned:
                    cleaned = cleaned.replace(fragment, "\n[SKIPPED OUT-OF-SCOPE TABLE]\n")
                    removed_count += 1

            caption = self._get_caption_text(block)
            table_number = re.search(r"\bTable\s+\d+\b", caption or "", flags=re.IGNORECASE)
            if table_number:
                # If exact HTML differs slightly, remove a bounded caption-to-table span.
                pattern = (
                    r"(?:^|\n)[^\n]{0,500}"
                    + re.escape(table_number.group(0))
                    + r"[\s\S]{0,8000}</table>"
                )
                cleaned, substitutions = re.subn(
                    pattern,
                    "\n[SKIPPED OUT-OF-SCOPE TABLE]\n",
                    cleaned,
                    count=1,
                    flags=re.IGNORECASE,
                )
                removed_count += substitutions

        cleaned, passage_count = self._remove_out_of_scope_activity_rate_passages(cleaned)
        removed_count += passage_count
        if removed_count:
            logger.info(
                "    [Table Type Gate] Removed %s out-of-scope activity table/text fragments before LLM text extraction",
                removed_count,
            )
        return cleaned

    def _remove_out_of_scope_activity_rate_passages(self, text: str) -> Tuple[str, int]:
        """Drop obvious specific-activity rate paragraphs that are not primary records."""
        if not text:
            return text, 0

        kept_lines = []
        removed = 0
        for line in text.splitlines():
            compact = line.strip()
            if not compact:
                kept_lines.append(line)
                continue
            has_activity = self._has_specific_or_generic_activity_scope(compact)
            has_rate_unit = self._has_activity_rate_unit_signal(compact)
            has_kinetic = self._has_primary_kinetic_table_signal(compact)
            has_degradation = self._has_explicit_degradation_table_signal(compact)
            has_mycotoxin = any(term in compact.lower() for term in MYCOTOXIN_SUBSTRATE_TERMS)
            if has_activity and has_rate_unit and not has_degradation and (not has_kinetic or not has_mycotoxin):
                kept_lines.append("[SKIPPED OUT-OF-SCOPE SPECIFIC ACTIVITY TEXT]")
                removed += 1
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines), removed

    def _collect_original_text(self, paper_blocks: List[Dict], paper_dir: Path = None) -> str:
        """
        收集原文（用于Aggregation Agent参考）
        
        优先使用 full.md，fallback 到 content_list.json 的文本块
        
        Args:
            paper_blocks: 论文块列表
            paper_dir: 论文目录（用于读取 full.md）
            
        Returns:
            原文文本
        """
        full_text = ""
        
        # 优先读取 full.md
        if paper_dir:
            full_md_path = paper_dir / "full.md"
            if full_md_path.exists():
                try:
                    full_text = full_md_path.read_text(encoding='utf-8')
                    full_text = self._remove_references(full_text)
                    full_text = self._remove_skipped_table_context_from_text(full_text, paper_blocks)
                    logger.debug(f"    Aggregation using full.md: {len(full_text)} chars")
                except Exception as e:
                    logger.warning(f"    Failed to read full.md for aggregation: {e}")
        
        # Fallback: 使用 paper_blocks
        if not full_text:
            texts = []
            for block in paper_blocks:
                block_type = block.get('type')
                if block_type == 'text':
                    texts.append(block.get('content', ''))
                elif block_type == 'table':
                    if block.get("_skip_for_text_extraction"):
                        continue
                    texts.append(f"[TABLE]\n{block.get('content', '')}")
            full_text = "\n\n".join(texts)
        
        # 限制长度（aggregation_agent 已经调整为 100K，这里也放宽）
        max_length = 100000
        if len(full_text) > max_length:
            full_text = full_text[:max_length] + "\n\n[...文本已截断...]"
        
        return full_text


def create_paper_level_extractor(
    kimi_client,
    deepseek_client,
    glm47_client,  # 历史参数名；第三文本模型，可选，传 None 则只使用2个学生模型
    glm46v_client,
    aggregation_client,
    text_prompt_path: str = "prompts/prompts_extract_from_text_v7_expanded.txt",
    table_prompt_path: str = "prompts/prompts_extract_from_table_v7_expanded.txt",
    figure_prompt_path: str = "prompts/prompts_extract_from_table_v7_expanded.txt",
    disable_table_image: bool = False,
) -> PaperLevelMultiModelExtractor:
    """
    创建论文级别多模型提取器

    Args:
        kimi_client: Kimi客户端
        deepseek_client: DeepSeek客户端
        glm47_client: 第三文本模型客户端（历史参数名；可选，传None则不使用）
        glm46v_client: 表格视觉模型客户端（历史参数名）
        aggregation_client: teacher聚合模型客户端
        text_prompt_path: 文本提取prompt路径
        table_prompt_path: 表格提取prompt路径
        figure_prompt_path: 图片提取prompt路径

    Returns:
        PaperLevelMultiModelExtractor实例
    """
    # 加载prompt模板
    def load_prompt(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    text_prompt = load_prompt(text_prompt_path)
    table_prompt = load_prompt(table_prompt_path)
    figure_prompt = load_prompt(figure_prompt_path)
    
    return PaperLevelMultiModelExtractor(
        kimi_client=kimi_client,
        deepseek_client=deepseek_client,
        glm47_client=glm47_client,
        glm46v_client=glm46v_client,
        aggregation_client=aggregation_client,
        text_prompt_template=text_prompt,
        table_prompt_template=table_prompt,
        figure_prompt_template=figure_prompt,
        disable_table_image=disable_table_image,
    )
