"""
Paper-Level Multi-Model Extractor with Aggregation Agent

使用文献中的思路：
1. 多个"学生"模型（Kimi, DeepSeek, 可选GLM-4.7）分别提取整篇论文
2. 一个"老师"模型（GPT-5.1/Claude 3.5）聚合结果

优势：
- 论文级别提取，可以对齐跨块的记录
- 利用强模型的推理能力智能聚合
- 一次调用得到最优结果

配置灵活性：
- 支持2个学生模型（Kimi + DeepSeek）+ GLM-4.6V（视觉）
- 支持3个学生模型（Kimi + DeepSeek + GLM-4.7）+ GLM-4.6V（视觉）
"""

import asyncio
import logging
import random
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
# GLM-4.6V 优化配置
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

# 优化2: 降低max_tokens
GLM46V_MAX_TOKENS = 4096  # 从默认的8192降低到4096

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
            logger.info(f"    [GLM-4.6V Stats] Images: {cls.total_images}, Tokens: {cls.total_tokens:,}, Avg: {avg_tokens:,}/img")
        if cls.skipped_tables > 0:
            logger.info(f"    [GLM-4.6V Stats] Skipped tables: {cls.skipped_tables} ({cls.skipped_tables/(cls.total_images+cls.skipped_tables)*100:.1f}%)")
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
    "glm-4.6v": 1,   # GLM多模态: 非常严格，1并发（API限制）
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
       - 组合A: Kimi (文本) + GLM-4.6V (表格图片，共享)
       - 组合B: DeepSeek (文本) + GLM-4.6V (表格图片，共享)
       - 组合C: GLM-4.7 (文本) + GLM-4.6V (表格图片，共享) [可选]
    2. 用Aggregation Agent（GPT-5.1）聚合N个结果

    灵活配置：
    - 2学生模型模式：kimi_client + deepseek_client（glm47_client=None）
    - 3学生模型模式：kimi_client + deepseek_client + glm47_client
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
        figure_prompt_template: str
    ):
        """
        Args:
            kimi_client: Kimi文本模型
            deepseek_client: DeepSeek文本模型
            glm47_client: GLM-4.7文本模型（可选，传None则不使用）
            glm46v_client: GLM-4.6V多模态模型
            aggregation_client: 聚合用的强模型（GPT-5.1）
            text_prompt_template: 文本提取prompt
            table_prompt_template: 表格提取prompt
            figure_prompt_template: 图片提取prompt
        """
        # 构建文本模型字典，自动过滤掉 None 的客户端
        self.text_models = {
            "kimi": kimi_client,
            "deepseek": deepseek_client,
        }
        if glm47_client is not None:
            self.text_models["glm-4.7"] = glm47_client

        self.multimodal_model = glm46v_client
        self.aggregation_client = aggregation_client

        self.text_prompt = text_prompt_template
        self.table_prompt = table_prompt_template
        self.figure_prompt = figure_prompt_template

        logger.info("Initialized PaperLevelMultiModelExtractor")
        logger.info(f"  - Text models: {list(self.text_models.keys())}")
        logger.info(f"  - Multimodal model: GLM-4.6V")
        logger.info(f"  - Aggregation model: GPT-5.1")
    
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
        
        # ================================================================
        # Step 0: 先用 GLM-4.6V 提取所有表格图片（只调用一次，结果共享给所有学生模型）
        # ================================================================
        table_blocks = [b for b in paper_blocks if b.get('type') == 'table']
        
        shared_table_records = []
        if table_blocks:
            logger.info(f"\n[Step 0/3] Extracting {len(table_blocks)} tables with GLM-4.6V (shared by all student models)...")
            shared_table_records = await self._extract_all_tables_once(
                table_blocks=table_blocks,
                paper_dir=paper_dir
            )
            logger.info(f"  ✓ GLM-4.6V extracted {len(shared_table_records)} table records (will be shared)")
        
        # Step 1: 用N个学生模型【并行】提取文本（不再重复调用 GLM-4.6V）
        # 🔧 使用Semaphore限制每个API的并发数 + 指数退避重试
        num_student_models = len(self.text_models)
        logger.info(f"\n[Step 1/3] Extracting TEXT with {num_student_models} student models (parallel, tables already extracted)...")
        
        # 并行创建所有模型的提取任务（只提取文本，表格结果共享）
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
                # 合并共享的表格结果
                all_records = records + [
                    {**r, '_source_model': model_name} for r in shared_table_records
                ]
                logger.info(f"  [{model_name.upper()}] ✓ Extracted {len(records)} text + {len(shared_table_records)} shared table records")
                return model_name, all_records
            except Exception as e:
                logger.error(f"  [{model_name.upper()}] ✗ Extraction failed: {e}")
                return model_name, []
        
        # 并行执行三个模型
        tasks = [
            extract_with_model(name, client) 
            for name, client in self.text_models.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果
        model_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"  Model extraction exception: {result}")
            else:
                model_name, records = result
                model_results[model_name] = records
        
        # Step 2: 用Aggregation Agent聚合结果
        logger.info(f"\n[Step 2/3] Aggregating results with GPT-5.1...")
        
        # 收集原文（用于Aggregation Agent参考）- 使用 full.md
        original_text = self._collect_original_text(paper_blocks, paper_dir)
        
        # 调用Aggregation Agent（支持工具调用）
        from src.agents.aggregation_agent import AggregationAgent
        
        aggregation_agent = AggregationAgent(
            llm_client=self.aggregation_client,
            model_name="GPT-5.1",
            paper_dir=paper_dir  # 🔥 传入paper_dir用于工具调用
        )
        
        aggregated_records = aggregation_agent.aggregate(
            original_text=original_text,
            model_results=model_results,
            doi=doi,
            paper_blocks=paper_blocks  # 🔥 传入paper_blocks用于查找表格
        )
        
        logger.info(f"  ✓ Final aggregated records: {len(aggregated_records)}")
        logger.info(f"{'='*80}\n")
        
        return {
            "aggregated_records": aggregated_records,
            "model_results": model_results,
            "doi": doi,
            "num_blocks": len(paper_blocks)
        }
    
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
        for block in table_blocks:
            if self._filter_table_by_caption(block):
                filtered_tables.append(block)

        total_tables = len(table_blocks)
        filtered_count = len(filtered_tables)
        skipped_count = total_tables - filtered_count

        logger.info(f"    [Smart Routing] Pre-filtering: {total_tables} → {filtered_count} (skipped: {skipped_count})")

        if not filtered_tables:
            return []

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
                # 纯文本提取
                text_only_tables.append((block_id, block))
                TokenTracker.add_text_only_table()
                logger.debug(f"    [Smart Routing] Table {block_id}: TEXT-ONLY ({reason})")
            else:
                # 视觉模型提取
                vision_model_tables.append((block_id, block))
                TokenTracker.add_vision_model_table()
                logger.debug(f"    [Smart Routing] Table {block_id}: VISION MODEL ({reason})")

        # 输出路由统计
        logger.info(f"    [Smart Routing] Text-only: {len(text_only_tables)}, Vision: {len(vision_model_tables)}, Skip: {len(no_keyword_tables)}")

        all_table_records = []

        # 步骤3a: 处理纯文本表格（并行）
        if text_only_tables:
            logger.debug(f"    [Smart Routing] Processing {len(text_only_tables)} tables with text-only extraction...")
            text_tasks = []
            for block_id, block in text_only_tables:
                task = self._extract_table_text_only(block, block_id)
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
            glm_semaphore = get_semaphore("glm-4.6v")

            vision_tasks = []
            for block_id, block in vision_model_tables:
                task = self._extract_with_semaphore(
                    glm_semaphore,
                    self._extract_table_block_multimodal,
                    self.multimodal_model,
                    block,
                    block_id,
                    paper_dir,
                    "glm-4.6v"
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
                        record['_extracted_by'] = 'glm-4.6v'
                        record['_extraction_method'] = extraction_type
                    all_table_records.extend(result)
                    logger.debug(f"    [Vision] Table {block_id}: {len(result)} records")

        logger.debug(f"    [Smart Routing] Total table records: {len(all_table_records)}")

        # 记录Token统计（每篇论文结束时输出）
        TokenTracker.log_stats()

        return all_table_records
    
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
        glm_semaphore = get_semaphore("glm-4.6v")
        
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
                    glm_semaphore,
                    self._extract_table_block_multimodal,
                    multimodal_model,
                    block,
                    block_id,
                    paper_dir,
                    model_name
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
            {"role": "system", "content": "You are an expert in enzyme kinetics data extraction. Extract ALL enzyme kinetic parameters from the given scientific article."},
            {"role": "user", "content": f"{self.text_prompt}\n\n=== ARTICLE FULL TEXT ===\n\n{merged_text}"}
        ]
        
        # 在线程池中运行同步调用
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: model.chat(messages=messages, temperature=0.1, task=f"text_{model_name}")
        )
        return self._parse_json_response(response)
    
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

        logger.debug(f"    [Smart Routing] Header lacks kinetic keywords, skipping table")
        return False

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

        # 步骤1: 检查表头是否含动力学关键词
        has_kinetic_keywords = self._check_table_headers_for_keywords(block)
        if not has_kinetic_keywords:
            return (None, "no_kinetic_keywords")

        # 步骤2: 检查HTML是否完整且可解析
        if not table_content or len(table_content) < 100:
            return (False, "html_incomplete")

        # 步骤3: 尝试用pandas解析
        if self._can_parse_html_with_pandas(table_content):
            return (True, "pandas_success")

        return (False, "pandas_failed")

    async def _extract_table_text_only(
        self,
        block: Dict,
        block_id: int
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
        # 选择一个文本模型进行提取（优先使用Kimi）
        text_model = self.text_models.get("kimi") or self.text_models.get("deepseek")
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

        # 构建提示词
        prompt = f"""{self.table_prompt}

=== 表格信息 ===

【标题】
{caption if caption else '(无标题)'}

【脚注】
{footnote if footnote else '(无脚注)'}

【表格HTML内容】
{table_content[:5000] if table_content else '(无HTML内容)'}

请从上述HTML表格中提取酶动力学数据。注意：
1. 优先使用HTML中的数值数据
2. 注意检查单位是否正确
3. 如果某个参数在HTML中没有，标记为 null"""

        # 使用文本模型提取
        semaphore = get_semaphore("kimi" if "kimi" in self.text_models else "deepseek")
        model_name = "kimi" if "kimi" in self.text_models else "deepseek"

        try:
            records = await self._extract_with_semaphore(
                semaphore,
                self._extract_text_block,
                text_model,
                prompt,
                block_id,
                model_name
            )
            logger.debug(f"    [Text-Only] Table {block_id}: extracted {len(records)} records")
            return records
        except Exception as e:
            logger.error(f"    [Text-Only] Table {block_id} extraction failed: {e}")
            return []

    async def _extract_text_block(
        self,
        model,
        content: str,
        block_id: int,
        model_name: str
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
            lambda: model.chat(messages=messages, temperature=0.1, task=f"text_{model_name}")
        )
        return self._parse_json_response(response)

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

    async def _extract_table_block_multimodal(
        self,
        model,
        block: Dict,
        block_id: int,
        paper_dir: Path,
        model_name: str
    ) -> List[Dict]:
        """
        使用多模态模型提取表格块（三源融合策略）
        
        融合三个信息源：
        1. 表格标题和脚注（语义上下文）
        2. HTML表格内容（结构化尝试）
        3. 表格图片（视觉真实值 - 主要来源）
        """
        import os
        
        # 提取三个信息源
        caption = block.get('table_caption', '')
        if isinstance(caption, list):
            caption = ' '.join(caption)
        
        footnote = block.get('table_footnote', '')
        if isinstance(footnote, list):
            footnote = ' '.join(footnote)
        
        table_content = block.get('table_body', '') or block.get('content', '')
        
        # 获取表格图片路径
        img_path = (
            block.get('img_path') or 
            block.get('image_path') or 
            block.get('table_img') or
            block.get('table_image')
        )
        
        full_image_path = None
        if img_path:
            full_image_path = str(paper_dir / img_path)
            if not os.path.exists(full_image_path):
                # 尝试在images子目录查找
                alt_path = str(paper_dir / 'images' / os.path.basename(img_path))
                if os.path.exists(alt_path):
                    full_image_path = alt_path
                else:
                    logger.warning(f"    Table image not found: {full_image_path}")
                    full_image_path = None
        
        # 构建提示词（三源融合）
        prompt = f"""{self.table_prompt}

=== 表格信息 ===

【标题】
{caption if caption else '(无标题)'}

【脚注】
{footnote if footnote else '(无脚注)'}

【表格HTML内容】
{table_content[:3000] if table_content else '(无HTML内容)'}

请综合上述三个信息源（标题、脚注、HTML、图片），交叉验证后提取数据：

**数据提取原则：**
1. **交叉验证**：优先使用图片和HTML内容一致的数据（可信度最高）
2. **冲突处理**：如果图片和HTML出现不一致，请仔细核对：
   - 检查HTML是否解析错误（如科学计数法、上下标）
   - 检查图片识别是否有误（如模糊字符、特殊符号）
   - 如果无法确定，优先采用图片数据，但在notes中标注"⚠️ HTML与图片数据不一致"
3. **准确性要求**：
   - 绝对不要臆造或推测数据
   - 特别注意单位（mM vs μM, s⁻¹ vs min⁻¹等）
   - 特别注意数量级（10³ vs 10⁶, 0.1 vs 1.0等）
4. **缺失数据**：如果某个参数在所有信息源中都没有，直接标记为 null，不要填充"""

        loop = asyncio.get_event_loop()

        if full_image_path and os.path.exists(full_image_path):
            # 优化3: 压缩图片
            compressed_image_path = self._compress_image_if_needed(full_image_path)

            # 优化2: 使用压缩后的图片路径
            image_to_use = compressed_image_path

            # 多模态模式：文本 + 图片
            messages = [
                {
                    "role": "user",
                    "text": prompt,
                    "image_path": image_to_use
                }
            ]
            logger.debug(f"    Block {block_id}: Using multimodal extraction with image: {image_to_use}")

            try:
                # 优化2: 降低max_tokens以减少token消耗
                response = await loop.run_in_executor(
                    None,
                    lambda: model.chat(messages=messages, is_multimodal=True, temperature=0.1, max_tokens=GLM46V_MAX_TOKENS, task="table_vision")
                )

                # 清理临时压缩文件
                if compressed_image_path != full_image_path and os.path.exists(compressed_image_path):
                    try:
                        os.remove(compressed_image_path)
                    except:
                        pass
            except Exception as e:
                logger.error(f"    Block {block_id}: Multimodal extraction failed: {e}")
                logger.error(f"    Image path: {full_image_path}")
                logger.error(f"    Prompt length: {len(prompt)} chars")
                raise
        else:
            # 纯文本模式（fallback）
            messages = [
                {"role": "system", "content": "You are an expert in enzyme kinetics data extraction."},
                {"role": "user", "content": prompt}
            ]
            logger.warning(f"    Block {block_id}: No table image, using text-only extraction")
            
            try:
                response = await loop.run_in_executor(
                    None, 
                    lambda: model.chat(messages=messages, temperature=0.1, task="table_text_only")
                )
            except Exception as e:
                logger.error(f"    Block {block_id}: Text-only extraction failed: {e}")
                logger.error(f"    Prompt length: {len(prompt)} chars")
                raise
        
        return self._parse_json_response(response)
    
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
        
        content = response
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
        
        # 提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return [data]
            return data
        except:
            return []
    
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
    glm47_client,  # 可选，传 None 则只使用2个学生模型
    glm46v_client,
    aggregation_client,
    text_prompt_path: str = "prompts/prompts_extract_from_text.txt",
    table_prompt_path: str = "prompts/prompts_extract_from_table.txt",
    figure_prompt_path: str = "prompts/prompts_extract_from_figure.txt"
) -> PaperLevelMultiModelExtractor:
    """
    创建论文级别多模型提取器

    Args:
        kimi_client: Kimi客户端
        deepseek_client: DeepSeek客户端
        glm47_client: GLM-4.7客户端（可选，传None则不使用）
        glm46v_client: GLM-4.6V客户端
        aggregation_client: GPT-5.1或Claude 3.5客户端
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
        figure_prompt_template=figure_prompt
    )
