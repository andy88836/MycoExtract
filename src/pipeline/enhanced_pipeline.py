"""
增强版提取流水线 - 支持并发处理和Multi-Agent协作

主要特性：
1. 并发处理多篇论文 (ThreadPoolExecutor)
2. DOI从metadata提取
3. 文本分块时的上下文重叠
4. 可选的Multi-Agent协作模式（Extractor-Reviewer-Synthesizer）
"""

import os
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass, asdict

from src.llm_extraction.text_extractor import TextExtractor
from src.llm_extraction.table_extractor import TableExtractor
# FigureExtractor disabled - figure data rarely contains kinetics parameters
# from src.llm_extraction.figure_extractor import FigureExtractor
from src.llm_extraction.enhanced_text_extractor import EnhancedTextExtractor
from src.llm_extraction.multi_agent_extractor import MultiAgentExtractor
from src.pipeline.content_filter import create_default_filter
from src.pipeline.post_processor import RecordMerger, ConditionExtractor, normalize_records_batch  # Entity alignment and data fusion
from src.utils.data_validator import DataValidator
from src.utils.quality_constraints import QualityConstraintFilter  # Quality constraint filtering
from src.utils.sequence_enricher import SequenceEnricher
from src.utils.quality_analyzer import QualityAnalyzer, analyze_extraction_results  # 质量分析替代HITL
from src.utils.logging_config import setup_logging, get_logger

# Setup logging to logs/ directory
logger = get_logger(__name__)


@dataclass
class PipelineStats:
    """流水线统计信息"""
    total_papers: int = 0
    processed_papers: int = 0
    failed_papers: int = 0
    skipped_papers: int = 0  # 预检查跳过的论文数
    total_records: int = 0
    text_records: int = 0
    table_records: int = 0
    figure_records: int = 0
    # 去重合并统计
    records_before_merge: int = 0
    records_after_merge: int = 0
    records_merged: int = 0
    # 时间统计
    total_time: float = 0.0
    avg_time_per_paper: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EnhancedExtractionPipeline:
    """
    增强版提取流水线
    
    新增功能：
    - 并发处理论文 (ThreadPoolExecutor)
    - 从metadata.json提取DOI
    - 文本分块时保留上下文重叠 (可选)
    - Multi-Agent团队协作模式 (可选)
    """
    
    def __init__(
        self,
        llm_client=None,
        text_client=None,
        multimodal_client=None,
        review_client=None,  # Add review_client for ReviewerAgent
        text_prompt_path: str = "prompts/prompts_extract_from_text.txt",
        table_prompt_path: str = "prompts/prompts_extract_from_table.txt",
        figure_prompt_path: str = "prompts/prompts_extract_from_figure.txt",
        max_workers: int = 5,
        context_overlap_sentences: int = 0,
        use_multi_agent: bool = False,
        max_retries: int = 2,
        save_intermediate: bool = False,  # 默认不保存每篇论文的单独文件（避免输出文件过多）
        enable_sequence_enrichment: bool = True,
        sequence_enrichment_threshold: float = 0.9,
        enable_figure_extraction: bool = False,  # Disabled by default - figures rarely contain kinetics data
        # 实体对齐与数据融合选项
        enable_record_merge: bool = True,  # 启用重复记录合并
        merge_km_tolerance: float = 0.1,   # Km值匹配容差 (0.1 = 10%)
        merge_prefer_table: bool = True,    # 合并时优先使用表格数据
        # Multi-Agent 审核选项
        enable_multiagent_review: bool = False,  # 启用 Multi-Agent 审核系统
        multiagent_enable_web: bool = True,      # 审核时启用网络验证
        # Agent流水线模式（新）
        use_agent_pipeline: bool = False,  # 使用Agent方式处理后处理步骤
        # 多模型投票选项（新）
        use_multi_model_voting: bool = False,  # 启用多模型投票提取
        kimi_client=None,                       # Kimi客户端
        deepseek_client=None,                   # DeepSeek客户端
        glm47_client=None,                      # GLM-4.7客户端
        glm46v_client=None,                     # GLM-4.6V客户端
        # Paper-level Aggregation选项（新）
        use_paper_level_aggregation: bool = False,  # 启用paper-level aggregation
        aggregation_client=None,                     # Aggregation Agent客户端（GPT-5.1）
        # 质量约束选项（v7.0 expanded）
        require_sequence: bool = True,        # 是否要求序列信息（v7.0可设为False接受固定化/粗酶）
        # 直接读取full.md选项（减少API调用）
        use_full_md: bool = True,             # 是否直接读取full.md文件（默认True，大幅减少API调用）
    ):
        """
        Args:
            llm_client: 统一LLM客户端（兼容旧代码，如果提供则用于所有任务）
            text_client: 文本提取专用客户端（DeepSeek等）
            multimodal_client: 多模态提取专用客户端（GLM-4.5V等，用于表格和图片）
            review_client: 审核专用客户端（GPT-5.1等，用于ReviewerAgent）
            text_prompt_path: 文本提取提示词路径
            table_prompt_path: 表格提取提示词路径
            figure_prompt_path: 图片提取提示词路径
            max_workers: 最大并发worker数（推荐3-10，取决于API限制）
            context_overlap_sentences: 分块时保留的上下文句子数（0=禁用）
            use_multi_agent: 是否启用Multi-Agent协作模式
            max_retries: 失败重试次数
            save_intermediate: 是否实时保存中间结果
        """
        # 支持新旧两种初始化方式
        if llm_client:
            # 旧方式：单一客户端
            self.text_client = llm_client
            self.multimodal_client = llm_client
            self.review_client = llm_client  # Fallback to llm_client if not provided
        else:
            # 新方式：分离的text和multimodal客户端
            self.text_client = text_client
            self.multimodal_client = multimodal_client
            self.review_client = review_client if review_client else text_client  # Default to text_client
        
        if not self.text_client or not self.multimodal_client:
            raise ValueError("Must provide either llm_client or both text_client and multimodal_client")
        
        self.text_prompt_path = text_prompt_path
        self.table_prompt_path = table_prompt_path
        self.figure_prompt_path = figure_prompt_path
        self.max_workers = max_workers
        self.context_overlap_sentences = context_overlap_sentences
        self.use_multi_agent = use_multi_agent
        self.max_retries = max_retries
        self.save_intermediate = save_intermediate
        
        # 线程安全的锁
        self._lock = Lock()
        
        # 统计信息
        self.stats = PipelineStats()
        
        # Figure extraction toggle (disabled by default)
        self.enable_figure_extraction = enable_figure_extraction
        
        # 内容过滤器
        self.content_filter = create_default_filter()
        
        # 质量约束过滤器 (新增：序列可获取性、霉菌毒素底物、解毒验证)
        self.quality_filter = QualityConstraintFilter(
            require_sequence=require_sequence,  # 可配置：v7.0 expanded设为False接受固定化/粗酶
            require_mycotoxin=True,     # 底物必须是霉菌毒素（保持不变）
            check_detoxification=True,  # 产物必须是解毒的（保持不变）
            strict_mode=False           # 宽松模式（允许部分不确定的记录通过）
        )
        
        # 序列富集器 (查询UniProt补充uniprot_id等信息)
        self.enable_sequence_enrichment = enable_sequence_enrichment
        self.sequence_enricher = SequenceEnricher(
            auto_fill_threshold=sequence_enrichment_threshold
        ) if enable_sequence_enrichment else None
        
        # 记录合并器 (实体对齐与数据融合)
        self.enable_record_merge = enable_record_merge
        self.record_merger = RecordMerger(
            km_tolerance=merge_km_tolerance,
            prefer_table_source=merge_prefer_table
        ) if enable_record_merge else None
        
        # Multi-Agent 审核系统
        self.enable_multiagent_review = enable_multiagent_review
        self.multiagent_enable_web = multiagent_enable_web
        
        # Agent流水线模式 - 使用新的 review_pipeline.py (含LLM审核 + 序列侦探)
        self.use_agent_pipeline = use_agent_pipeline
        self._review_pipeline = None

        # 直接读取full.md模式（大幅减少API调用）
        self.use_full_md = use_full_md
        if use_agent_pipeline:
            from src.agents.review_pipeline import PostExtractionPipeline
            self._review_pipeline = PostExtractionPipeline(
                llm_client=self.review_client,  # 用于ReviewerAgent的LLM审核（GPT-5.1）
                sequence_client=self.text_client,  # 用于SequenceDetective（GLM-4.6）
                config={
                    "enable_merge": enable_record_merge,
                    "enable_enrichment": enable_sequence_enrichment,
                    "km_tolerance": merge_km_tolerance,
                    "auto_fill_threshold": sequence_enrichment_threshold,
                    "enable_web_verification": multiagent_enable_web
                }
            )
        
        # 多模型投票模式 - 使用3个文本模型+1个多模态模型
        self.use_multi_model_voting = use_multi_model_voting
        self.use_paper_level_aggregation = use_paper_level_aggregation
        self.aggregation_client = aggregation_client
        self.multi_model_extractor = None
        self.paper_level_extractor = None
        
        if use_multi_model_voting:
            # 验证所需的客户端都已提供
            if not all([kimi_client, deepseek_client, glm47_client]):
                raise ValueError(
                    "Multi-model voting requires kimi_client, deepseek_client, and glm47_client. "
                    "Please provide all three text model clients."
                )
            
            from src.extractors.sync_multi_model_extractor import create_sync_multi_model_extractor
            
            self.multi_model_extractor = create_sync_multi_model_extractor(
                kimi_client=kimi_client,
                deepseek_client=deepseek_client,
                glm47_client=glm47_client,
                glm46v_client=glm46v_client or multimodal_client
            )
            
            logger.info("✓ Multi-model voting extractor initialized")
            logger.info("  - Text models: Kimi (moonshot-v1-128k), DeepSeek (deepseek-chat), GLM-4.7")
            logger.info("  - Multimodal model: GLM-4.6V")
            logger.info("  - Voting: 3/3 = high confidence, 2/3 = medium, 1/1/1 = low")
        
        # Paper-level Aggregation模式 - 使用2或3个文本模型提取+GPT-5.1聚合
        if use_paper_level_aggregation:
            # 验证所需的客户端都已提供（glm47_client可选）
            required_clients = [kimi_client, deepseek_client, aggregation_client]
            if not all(required_clients):
                raise ValueError(
                    "Paper-level aggregation requires kimi_client, deepseek_client, and aggregation_client. "
                    "glm47_client is optional (can be None for 2-student mode)."
                )

            from src.extractors.paper_level_extractor import create_paper_level_extractor

            self.paper_level_extractor = create_paper_level_extractor(
                kimi_client=kimi_client,
                deepseek_client=deepseek_client,
                glm47_client=glm47_client,  # 可以是None
                glm46v_client=glm46v_client or multimodal_client,
                aggregation_client=aggregation_client
            )

            # 根据glm47_client是否提供显示不同的日志
            if glm47_client:
                logger.info("✓ Paper-level aggregation extractor initialized (3-student mode)")
                logger.info("  - Student models: Kimi, DeepSeek, GLM-4.7")
            else:
                logger.info("✓ Paper-level aggregation extractor initialized (2-student mode)")
                logger.info("  - Student models: Kimi, DeepSeek")
            logger.info("  - Multimodal model: GLM-4.6V")
            logger.info("  - Teacher model: GPT-5.1 (Aggregation Agent)")
        
        logger.info(f"✓ Enhanced Pipeline initialized")
        logger.info(f"  - Max workers: {max_workers}")
        logger.info(f"  - Context overlap: {context_overlap_sentences} sentences")
        logger.info(f"  - Multi-Agent mode: {'ON' if use_multi_agent else 'OFF'}")
        logger.info(f"  - Full.md mode: {'ON (1 API/paper)' if use_full_md else 'OFF (block-by-block)'}")
        logger.info(f"  - Figure extraction: {'ON' if enable_figure_extraction else 'OFF (disabled)'}")
        logger.info(f"  - Record merge: {'ON' if enable_record_merge else 'OFF'}")
        logger.info(f"  - Multi-Agent review: {'ON' if enable_multiagent_review else 'OFF'}")
        logger.info(f"  - Agent pipeline: {'ON' if use_agent_pipeline else 'OFF'}")
        logger.info(f"  - Multi-model voting: {'ON' if use_multi_model_voting else 'OFF'}")
        logger.info(f"  - Paper-level aggregation: {'ON' if use_paper_level_aggregation else 'OFF'}")
        logger.info(f"  - Max retries: {max_retries}")
        logger.info(f"  - Save intermediate: {save_intermediate}")
    
    def _load_prompt_template(self, prompt_path: str) -> str:
        """
        从文件加载提示词模板
        
        Args:
            prompt_path: 提示词文件路径
            
        Returns:
            提示词模板字符串
        """
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
    
    def run(
        self,
        paper_dirs: List[str],
        output_dir: str = "results",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        并发处理多篇论文
        
        Args:
            paper_dirs: 论文目录列表
            output_dir: 输出目录
            progress_callback: 进度回调函数 callback(current, total, paper_name)
            
        Returns:
            {
                "results": {paper_name: records_list},
                "statistics": PipelineStats,
                "failed_papers": {paper_name: error_message}
            }
        """
        start_time = time.time()
        self.stats.total_papers = len(paper_dirs)
        
        logger.info("=" * 80)
        logger.info(f"🚀 Enhanced Pipeline Starting")
        logger.info(f"   Papers: {len(paper_dirs)}")
        logger.info(f"   Workers: {self.max_workers}")
        logger.info(f"   Multi-Agent: {'ON' if self.use_multi_agent else 'OFF'}")
        logger.info("=" * 80)
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = Path(output_dir)

        # === Phase 0: 论文级别预检查（快速跳过不相关论文） ===
        logger.info("\n" + "=" * 80)
        logger.info("🔍 Phase 0: Paper-Level Pre-Check (Token Optimization)")
        logger.info("=" * 80)

        from src.pipeline.paper_level_prechecker import PaperLevelPrechecker
        prechecker = PaperLevelPrechecker(
            min_mycotoxin_hits=1,
            min_kinetics_hits=2,
            enable_unit_check=True
        )

        # 转换为 Path 对象列表
        paper_dir_paths = [Path(pd) if isinstance(pd, str) else pd for pd in paper_dirs]

        # 预检查所有论文
        precheck_result = prechecker.batch_check_papers(paper_dir_paths)

        # 只保留通过预检查的论文
        valid_paper_dirs = [
            str(pd) for pd in paper_dir_paths
            if not precheck_result['results'][pd.name]['should_skip']
        ]

        # 更新统计
        self.stats.total_papers = len(paper_dir_paths)
        self.stats.skipped_papers = precheck_result['skipped']
        self.stats.processed_papers = 0  # 重置，后续会重新计数

        logger.info(f"  ✓ Pre-check completed: {len(valid_paper_dirs)}/{len(paper_dir_paths)} papers passed")

        if len(valid_paper_dirs) == 0:
            logger.warning("  ⚠️ All papers were skipped! No papers to process.")
            return {
                "results": {},
                "statistics": self.stats.to_dict(),
                "failed_papers": {},
                "precheck_stats": precheck_result
            }

        paper_dirs = valid_paper_dirs

        all_results = {}
        failed_papers = {}
        
        # 使用ThreadPoolExecutor并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self._process_paper_with_retry, paper_dir): paper_dir
                for paper_dir in paper_dirs
            }
            
            # 收集结果
            for future in as_completed(future_to_paper):
                paper_dir = future_to_paper[future]
                paper_name = Path(paper_dir).name
                
                try:
                    result = future.result()
                    
                    with self._lock:
                        if result['success']:
                            all_results[paper_name] = result['records']
                            self.stats.processed_papers += 1
                            self.stats.total_records += result['stats']['total']
                            self.stats.text_records += result['stats']['text']
                            self.stats.table_records += result['stats']['table']
                            self.stats.figure_records += result['stats']['figure']
                            
                            # 累计合并统计
                            merge_stats = result['stats'].get('merge_stats')
                            if merge_stats:
                                self.stats.records_before_merge += merge_stats['before']
                                self.stats.records_after_merge += merge_stats['after']
                                self.stats.records_merged += merge_stats['merged']
                            
                            # 构建日志信息
                            log_parts = [
                                f"✅ [{self.stats.processed_papers}/{self.stats.total_papers}] ",
                                f"{paper_name}: {result['stats']['total']} records ",
                                f"(T:{result['stats']['text']}, ",
                                f"TB:{result['stats']['table']}, ",
                                f"F:{result['stats']['figure']}"
                            ]
                            if merge_stats and merge_stats['merged'] > 0:
                                log_parts.append(f", M:-{merge_stats['merged']}")
                            log_parts.append(")")
                            logger.info("".join(log_parts))
                            
                            # 实时保存中间结果
                            if self.save_intermediate:
                                self._save_paper_result(paper_name, result['records'])
                        else:
                            failed_papers[paper_name] = result['error']
                            self.stats.failed_papers += 1
                            logger.error(
                                f"❌ [{self.stats.processed_papers + self.stats.failed_papers}/"
                                f"{self.stats.total_papers}] {paper_name}: {result['error']}"
                            )
                        
                        # 进度回调
                        if progress_callback:
                            progress_callback(
                                self.stats.processed_papers + self.stats.failed_papers,
                                self.stats.total_papers,
                                paper_name
                            )
                
                except Exception as e:
                    with self._lock:
                        error_msg = f"Exception during processing: {str(e)}"
                        failed_papers[paper_name] = error_msg
                        self.stats.failed_papers += 1
                        logger.error(f"❌ {paper_name}: {error_msg}")
                        logger.debug(traceback.format_exc())
        
        # 计算统计信息
        self.stats.total_time = time.time() - start_time
        if self.stats.processed_papers > 0:
            self.stats.avg_time_per_paper = self.stats.total_time / self.stats.processed_papers
        
        # 保存最终结果
        self._save_final_results(all_results, failed_papers, output_dir)
        
        # 打印总结
        self._print_summary()
        
        return {
            "results": all_results,
            "statistics": self.stats.to_dict(),
            "failed_papers": failed_papers
        }
    
    def _process_paper_with_retry(self, paper_dir: str) -> Dict[str, Any]:
        """
        处理单篇论文（带重试机制）
        
        Args:
            paper_dir: 论文目录路径
            
        Returns:
            {
                "success": bool,
                "records": List[Dict],
                "stats": {"total": int, "text": int, "table": int, "figure": int},
                "error": str (if success=False)
            }
        """
        paper_name = Path(paper_dir).name
        
        for attempt in range(self.max_retries + 1):
            try:
                return self._process_paper(paper_dir)
            
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"⚠️ {paper_name}: Attempt {attempt + 1} failed, retrying... ({str(e)})"
                    )
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    error_msg = f"Failed after {self.max_retries + 1} attempts: {str(e)}"
                    logger.error(f"❌ {paper_name}: {error_msg}")
                    logger.debug(traceback.format_exc())
                    return {
                        "success": False,
                        "records": [],
                        "stats": {"total": 0, "text": 0, "table": 0, "figure": 0},
                        "error": error_msg
                    }
    
    def _process_paper(self, paper_dir: str) -> Dict[str, Any]:
        """
        处理单篇论文
        
        Args:
            paper_dir: 论文目录路径
            
        Returns:
            处理结果字典
        """
        paper_dir = Path(paper_dir)
        paper_name = paper_dir.name
        
        # 1. 提取DOI
        doi = self._extract_doi(paper_dir)
        
        # 2. 读取content_list.json
        content_file = list(paper_dir.glob("*_content_list.json"))
        if not content_file:
            return {
                "success": False,
                "records": [],
                "stats": {"total": 0, "text": 0, "table": 0, "figure": 0},
                "error": "content_list.json not found"
            }
        
        with open(content_file[0], 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        
        # 3. 分类blocks (添加索引作为block_id)
        text_blocks = [(i, b) for i, b in enumerate(content_list) if b['type'] == 'text']
        table_blocks = [(i, b) for i, b in enumerate(content_list) if b['type'] == 'table']
        # Figure extraction disabled by default (rarely contains kinetics data)
        figure_blocks = []
        if self.enable_figure_extraction:
            # Support both 'figure' and 'image' types (different parsers use different naming)
            figure_blocks = [(i, b) for i, b in enumerate(content_list) if b['type'] in ('figure', 'image')]
        
        raw_records = []

        # 4. 提取文本（优先使用full.md，大幅减少API调用）
        text_records = []
        if self.use_full_md:
            # 尝试直接读取full.md文件
            full_md_path = paper_dir / "full.md"
            if full_md_path.exists():
                logger.info(f"  📄 Using full.md for text extraction (1 API call)...")
                text_records = self._extract_from_full_md(full_md_path, doi)
                raw_records.extend(text_records)
            else:
                logger.warning(f"  ⚠️ full.md not found, falling back to text blocks...")
                if text_blocks:
                    text_records = self._extract_from_text(text_blocks, doi)
                    raw_records.extend(text_records)
        elif text_blocks:
            text_records = self._extract_from_text(text_blocks, doi)
            raw_records.extend(text_records)
        
        # 5. 提取表格数据
        table_records = []
        figure_records = []
        
        if table_blocks:
            try:
                table_records = self._extract_from_table(table_blocks, doi, paper_dir)
                raw_records.extend(table_records)
            except Exception as e:
                logger.error(f"  Error in table extraction: {e}")
        
        # 6. 提取图表数据（默认禁用）
        if self.enable_figure_extraction and figure_blocks:
            try:
                figure_records = self._extract_from_figure(figure_blocks, doi, paper_dir)
                raw_records.extend(figure_records)
            except Exception as e:
                logger.error(f"  Error in figure extraction: {e}")
        
        # ===== 后处理阶段 =====
        # 可选：使用Paper-level Aggregation、Agent流水线或传统方式处理
        
        if self.use_paper_level_aggregation and self.paper_level_extractor:
            # ========== Paper-level Aggregation模式 (3个学生模型 + GPT-5.1聚合) ==========
            logger.info(f"  Using Paper-level Aggregation...")
            
            import asyncio
            
            # 使用paper-level extractor重新提取整篇论文
            result = asyncio.run(self.paper_level_extractor.extract_paper(
                paper_blocks=content_list,
                doi=doi,
                paper_dir=paper_dir
            ))
            
            all_records = result['aggregated_records']
            logger.info(f"  ✓ Aggregation completed: {len(all_records)} records")
            
            # 清理重复字段（Aggregation 可能产生的字段名不一致问题）
            if all_records:
                logger.info(f"  Cleaning duplicate fields...")
                all_records = self._clean_duplicate_fields(all_records)
            
            # 序列富集 (在返回前执行)
            enrichment_stats = None
            if self.enable_sequence_enrichment and self.sequence_enricher and all_records:
                logger.info(f"  Enriching sequences and substrates via UniProt/PubChem...")
                all_records, enrichment_stats = self.sequence_enricher.enrich_records(
                    all_records, 
                    auto_fill=True, 
                    verbose=False
                )
                logger.info(f"  Enrichment: {enrichment_stats.get('auto_filled', 0)} auto-filled, "
                            f"{enrichment_stats.get('candidates_found', 0)} candidates, "
                            f"{enrichment_stats.get('no_match', 0)} no match")
            
            # Schema规范化
            if all_records:
                logger.info(f"  Normalizing record schema...")
                all_records = normalize_records_batch(all_records)
                logger.info(f"  Schema normalized for {len(all_records)} records")
            
            # 统计信息
            stats = {
                "total": len(all_records),
                "text": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'text']),
                "table": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'table']),
                "figure": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'figure']),
                "merged": 0,  # Aggregation doesn't merge, it aggregates
                "model_results": result.get('model_results', {}),
                "enrichment": enrichment_stats
            }
            
            return {
                "success": True,
                "records": all_records,
                "stats": stats,
                "error": None
            }
        
        elif self.use_agent_pipeline and self._review_pipeline:
            # ========== Agent流水线模式 (带LLM审核) ==========
            # 使用 PostExtractionPipeline:
            #   1. DataProcessor - 数据清洗、合并、规范化 (规则处理，不用LLM)
            #   2. SequenceDetective - 蛋白质序列智能检索 (GLM-4.6 + UniProt API)
            #   3. ReviewerAgent - 智能审核 (GPT-5.1 + 联网验证工具)
            logger.info(f"  Using Agent Pipeline with LLM review...")
            
            import asyncio
            
            # 构建论文全文（用于SequenceDetective提取生物指纹）
            paper_text = '\n'.join(
                self._get_block_text(b) for _, b in text_blocks if self._get_block_text(b)
            ) if text_blocks else ""
            
            # 运行异步流水线
            result = asyncio.run(self._review_pipeline.process_async(
                records=raw_records,
                doi=doi,
                skip_review=not self.enable_multiagent_review,
                paper_text=paper_text,
                content_list=content_list
            ))
            
            all_records = result["records"]
            processing_stats = result.get("processing_stats", {})
            # 标准化merge_stats格式（适配Agent Pipeline返回的格式）
            merge_stats = {
                'before': processing_stats.get("input_count", len(raw_records)),
                'after': processing_stats.get("output_count", len(all_records)),
                'merged': processing_stats.get("input_count", len(raw_records)) - processing_stats.get("output_count", len(all_records))
            }
            review_summary = result.get("review_result")
        
        else:
            # ========== 传统处理模式 ==========
            # 7. 验证和清理数据 (移除额外字段、处理Unknown、计算置信度)
            logger.info(f"  Validating {len(raw_records)} raw records...")
            all_records = DataValidator.validate_batch(raw_records)
            logger.info(f"  After validation: {len(all_records)} valid records")
            
            # 7.2 质量约束过滤 (新增：序列可获取性、霉菌毒素底物、解毒验证)
            quality_filter_stats = None
            if all_records:
                logger.info(f"  Applying quality constraints (sequence/mycotoxin/detoxification)...")
                all_records, quality_filter_stats = self.quality_filter.filter_records(all_records)
                logger.info(f"  After quality filtering: {len(all_records)} records "
                           f"({quality_filter_stats['rejected']} rejected, "
                           f"{quality_filter_stats['rejection_rate']:.1f}% rejection rate)")
                if quality_filter_stats['rejected'] > 0:
                    logger.info(f"    Reasons: no_sequence={quality_filter_stats['rejected_no_sequence']}, "
                               f"non_mycotoxin={quality_filter_stats['rejected_non_mycotoxin']}, "
                               f"bioactivation={quality_filter_stats['rejected_bioactivation']}")
            
            # 7.3 合并同一论文中相同酶的信息 (补全 enzyme_full_name, organism 等字段)
            if all_records:
                logger.info(f"  Merging enzyme info within paper...")
                all_records = self._merge_enzyme_info_within_paper(all_records)
            
            # 7.5 使用 LLM 从原文本中补全温度/pH条件 (解决表格数据缺少条件信息的问题)
            if all_records and text_blocks:
                # 合并所有文本块用于条件提取
                full_text = '\n'.join(
                    self._get_block_text(b) for b in text_blocks if self._get_block_text(b)
                )
                if full_text:
                    logger.info(f"  Filling missing conditions (pH/temperature) via LLM...")
                    all_records = ConditionExtractor.fill_conditions_to_records(
                        all_records, full_text, llm_client=self.text_client
                    )
            
            # 8. 实体对齐与数据融合 (合并重复记录)
            records_before_merge = len(all_records)
            merge_stats = None
            if self.enable_record_merge and self.record_merger and all_records:
                logger.info(f"  Merging duplicate records (Entity Alignment)...")
                all_records = self.record_merger.merge_records(all_records)
                records_after_merge = len(all_records)
                merge_stats = {
                    'before': records_before_merge,
                    'after': records_after_merge,
                    'merged': records_before_merge - records_after_merge
                }
                if merge_stats['merged'] > 0:
                    logger.info(f"  Merged: {records_before_merge} -> {records_after_merge} "
                               f"(-{merge_stats['merged']} duplicates, "
                               f"{merge_stats['merged']/records_before_merge*100:.1f}% reduced)")
                else:
                    logger.info(f"  No duplicates found to merge")
            
            # 9. 序列富集 (查询UniProt补充uniprot_id等信息)
            enrichment_stats = None
            if self.enable_sequence_enrichment and self.sequence_enricher and all_records:
                logger.info(f"  Enriching sequences via UniProt...")
                all_records, enrichment_stats = self.sequence_enricher.enrich_records(
                    all_records, 
                    auto_fill=True, 
                    verbose=False
                )
                logger.info(f"  Enrichment: {enrichment_stats.get('auto_filled', 0)} auto-filled, "
                            f"{enrichment_stats.get('candidates_found', 0)} candidates, "
                            f"{enrichment_stats.get('no_match', 0)} no match")
            
            # 9.5 Schema规范化 (确保所有记录有完整字段)
            if all_records:
                logger.info(f"  Normalizing record schema...")
                all_records = normalize_records_batch(all_records)
                logger.info(f"  Schema normalized for {len(all_records)} records")
            
            # 9.8 Multi-Agent 审核
            review_summary = None
            if self.enable_multiagent_review and all_records:
                logger.info(f"  Multi-Agent review system starting...")
                # 传入content_list让Review Agent根据block_id智能提取相关段落
                all_records, review_summary = self._run_multiagent_review(all_records, doi, content_list)
                if review_summary:
                    logger.info(f"    Review: {review_summary.get('status_breakdown', {})}")
                    logger.info(f"    Confidence: {review_summary.get('confidence', {})}")
        
        # 10. 数据质量摘要
        quality_summary = DataValidator.get_quality_summary(all_records)
        logger.info(f"  Quality: Avg confidence={quality_summary['avg_confidence']:.3f}, "
                   f"High={quality_summary['high_quality']}, "
                   f"Med={quality_summary['medium_quality']}, "
                   f"Low={quality_summary['low_quality']}")
        
        # 11. 统计
        stats = {
            "total": len(all_records),
            "text": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'text']),
            "table": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'table']),
            "figure": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'figure']),
            "merged": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'merged']),
            "quality": quality_summary,
            "merge_stats": merge_stats,  # 合并统计信息
            "review_summary": review_summary  # Multi-Agent 审核摘要
        }
        
        return {
            "success": True,
            "records": all_records,
            "stats": stats,
            "error": None
        }
    
    def _get_block_text(self, block: Dict) -> str:
        """
        从块中提取文本内容
        
        Args:
            block: 内容块（可能是tuple或dict）
            
        Returns:
            文本字符串
        """
        # 如果是tuple (block_id, block_dict)
        if isinstance(block, tuple) and len(block) == 2:
            block = block[1]
        
        if isinstance(block, str):
            return block
        
        if not isinstance(block, dict):
            return ''
        
        # 尝试多个可能的字段名
        for field in ['text', 'content', 'body', 'para']:
            if field in block:
                text = block[field]
                if isinstance(text, str):
                    return text.strip()
                elif isinstance(text, list):
                    return ' '.join(str(t) for t in text).strip()
        
        return ''

    def _extract_doi(self, paper_dir: Path) -> str:
        """
        提取DOI
        优先级:
        1. 从metadata.json读取
        2. 从content_list.json中的文本提取（首页DOI链接）
        3. 从目录名提取（备用）
        
        Args:
            paper_dir: 论文目录
            
        Returns:
            DOI字符串（格式如: 10.3390/ijms25126455）
        """
        import re
        
        # 1. 尝试从metadata.json提取
        metadata_file = paper_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    doi = metadata.get('doi')
                    if doi:
                        logger.debug(f"DOI from metadata.json: {doi}")
                        return doi
            except Exception as e:
                logger.warning(f"Failed to read metadata.json: {e}")
        
        # 2. 从content_list.json中提取DOI（首页文本）
        content_file = list(paper_dir.glob("*_content_list.json"))
        if content_file:
            try:
                with open(content_file[0], 'r', encoding='utf-8') as f:
                    content_list = json.load(f)
                
                # DOI正则表达式（匹配标准DOI格式）
                # 处理空格问题：https:/ /doi.org/10.3390/ijms25126455
                # 例如: 10.3390/ijms25126455, 10.1016/j.heliyon.2022.e10217
                # 先移除文本中的换行和多余空格
                doi_pattern = r'(?:doi\.org\s*/\s*|DOI\s*:?\s*)(10\.\d{4,}/[^\s,\])"\']+)'
                
                # 只搜索前50个文本块（DOI通常在首页，但有些PDF解析可能会分散）
                for i, block in enumerate(content_list[:50]):
                    if block.get('type') in ('text', 'discarded'):
                        text = block.get('text', '')
                        # 跳过引用文献（通常包含参考文献编号或"References"等）
                        if re.search(r'^\s*\[\d+\]|^References|^Bibliography', text, re.IGNORECASE):
                            continue
                        
                        # 清理文本中的空格（处理 "https:/ /doi.org/" 这种情况）
                        cleaned_text = re.sub(r'\s+', ' ', text)
                        
                        matches = re.findall(doi_pattern, cleaned_text, re.IGNORECASE)
                        if matches:
                            # 清理DOI（去除尾部标点和空格）
                            doi = matches[0].rstrip('.,;) ')
                            # 验证DOI格式
                            if re.match(r'^10\.\d{4,}/', doi):
                                logger.info(f"  DOI extracted from content: {doi}")
                                return doi
            except Exception as e:
                logger.warning(f"Failed to extract DOI from content_list: {e}")
        
        # 3. 从目录名提取（备用）
        doi = paper_dir.name
        logger.warning(f"  Using folder name as DOI (fallback): {doi}")
        return doi
    
    def _extract_from_text(self, text_blocks: List[Tuple[int, Dict]], doi: str) -> List[Dict]:
        """
        从文本提取
        
        Args:
            text_blocks: 文本块列表 [(block_id, block), ...]
            doi: DOI
            
        Returns:
            提取的记录列表
        """
        # 预过滤文本块（去除参考文献、摘要等）
        logger.info(f"  Filtering {len(text_blocks)} text blocks...")
        # 只传递block内容进行过滤，保留block_id
        blocks_only = [b for _, b in text_blocks]
        filtered_blocks = self.content_filter.filter_text_blocks(blocks_only)
        # 重新匹配block_id
        filtered_with_ids = []
        for block_id, block in text_blocks:
            if block in filtered_blocks:
                filtered_with_ids.append((block_id, block))
        logger.info(f"  After filtering: {len(filtered_with_ids)} blocks remain")
        
        if not filtered_with_ids:
            logger.warning("  ⚠️ No text blocks after filtering")
            return []
        
        # 提取blocks和ids
        block_ids = [bid for bid, _ in filtered_with_ids]
        blocks_only = [b for _, b in filtered_with_ids]
        
        # 执行提取
        if self.use_multi_model_voting:
            # 多模型投票模式 - 使用3个文本模型并行提取+投票
            logger.info("  Using multi-model voting extraction (3 models)...")
            
            all_records = []
            for block_id, block in filtered_with_ids:
                text_content = block.get('text', '')
                
                # 使用多模型提取器
                records = self.multi_model_extractor.extract_from_text(
                    text_content=text_content,
                    prompt_template=self._load_prompt_template(self.text_prompt_path),
                    doi=doi,
                    block_id=block_id
                )
                
                all_records.extend(records)
            
            logger.info(f"  Multi-model voting extracted {len(all_records)} records from text")
            records = all_records
            
        elif self.use_multi_agent:
            # Multi-Agent模式
            extractor = MultiAgentExtractor(
                llm_client=self.text_client,  # 使用text_client
                prompt_paths={
                    "text": self.text_prompt_path,
                    "table": self.table_prompt_path,
                    "figure": self.figure_prompt_path
                },
                source_type="text"
            )
            records = extractor.extract(blocks_only, doi=doi)
        else:
            # 标准模式
            if self.context_overlap_sentences > 0:
                # 使用带overlap的增强提取器
                extractor = EnhancedTextExtractor(
                    llm_client=self.text_client,  # 使用text_client
                    prompt_path=self.text_prompt_path,
                    overlap_sentences=self.context_overlap_sentences
                )
            else:
                # 使用标准提取器
                extractor = TextExtractor(
                    llm_client=self.text_client,  # 使用text_client
                    prompt_path=self.text_prompt_path
                )
            
            records = extractor.extract(blocks_only, doi=doi)
        
        # 为每条记录添加block_id溯源
        # 简化假设：每个block平均产生相同数量的记录，按顺序分配
        if records and block_ids:
            records_per_block = max(1, len(records) // len(block_ids))
            for i, record in enumerate(records):
                block_index = min(i // records_per_block, len(block_ids) - 1)
                block_id = f"text_block_{block_ids[block_index]}"
                if "source_in_document" in record and isinstance(record["source_in_document"], dict):
                    record["source_in_document"]["block_id"] = block_id
                else:
                    record["block_id"] = block_id
        
        return records

    def _extract_from_full_md(self, full_md_path: Path, doi: str) -> List[Dict]:
        """
        直接从full.md文件提取（大幅减少API调用）

        Args:
            full_md_path: full.md文件路径
            doi: DOI

        Returns:
            提取的记录列表
        """
        try:
            # 读取full.md
            full_text = full_md_path.read_text(encoding='utf-8')
            original_len = len(full_text)

            # 去除References部分
            full_text = self._remove_references(full_text)
            logger.info(f"  📄 Read full.md: {original_len} chars -> {len(full_text)} chars (after removing references)")

            # 构造一个虚拟的block对象，复用现有的提取器
            virtual_block = {
                'type': 'text',
                'text': full_text,
                'block_id': 'full_md'
            }

            # 使用标准文本提取器
            extractor = TextExtractor(
                llm_client=self.text_client,
                prompt_path=self.text_prompt_path
            )

            records = extractor.extract([virtual_block], doi=doi)

            # 为记录添加源标识
            for record in records:
                if "source_in_document" in record and isinstance(record["source_in_document"], dict):
                    record["source_in_document"]["block_id"] = "full.md"
                    record["source_in_document"]["extraction_mode"] = "full_document"
                else:
                    record["block_id"] = "full.md"
                    record["extraction_mode"] = "full_document"

            logger.info(f"  ✓ Extracted {len(records)} records from full.md")
            return records

        except Exception as e:
            logger.error(f"  ❌ Failed to extract from full.md: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def _remove_references(self, text: str) -> str:
        """
        从文本中移除References部分

        Args:
            text: 原始文本

        Returns:
            移除References后的文本
        """
        # 常见的References标题模式
        import re
        patterns = [
            r'\nReferences\s*\n',
            r'\nReferences\s*$',
            r'\n参考文献\s*\n',
            r'\nREFERENCES\s*\n',
            r'\nReferences\s*\r?\n',  # 跨行处理
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return text[:match.start()]

        return text

    def _extract_from_table(self, table_blocks: List[Tuple[int, Dict]], doi: str, paper_dir: Path) -> List[Dict]:
        """
        从表格提取
        
        Args:
            table_blocks: 表格块列表 [(block_id, block), ...]
            doi: DOI
            paper_dir: 论文目录（用于解析图像路径）
            
        Returns:
            提取的记录列表
        """
        # 提取block_ids和blocks
        block_ids = [bid for bid, _ in table_blocks]
        blocks_only = [b for _, b in table_blocks]
        
        # 添加图像路径
        for block in blocks_only:
            if 'image_path' in block:
                block['image_path'] = str(paper_dir / block['image_path'])
        
        # 执行提取
        if self.use_multi_model_voting:
            # 多模型投票模式 - 表格也用文本模型提取+投票
            logger.info("  Using multi-model voting for tables (3 models)...")
            
            all_records = []
            for block_id, block in table_blocks:
                # 获取表格内容（优先使用table_body，fallback到html）
                table_content = block.get('table_body') or block.get('html', '')
                
                # 使用多模型提取器
                records = self.multi_model_extractor.extract_from_table(
                    table_content=table_content,
                    prompt_template=self._load_prompt_template(self.table_prompt_path),
                    doi=doi,
                    block_id=block_id
                )
                
                all_records.extend(records)
            
            logger.info(f"  Multi-model voting extracted {len(all_records)} records from tables")
            records = all_records
            
        elif self.use_multi_agent:
            # Multi-Agent模式
            extractor = MultiAgentExtractor(
                llm_client=self.multimodal_client,  # 使用multimodal_client
                prompt_paths={
                    "text": self.text_prompt_path,
                    "table": self.table_prompt_path,
                    "figure": self.figure_prompt_path
                },
                source_type="table"
            )
            records = extractor.extract(blocks_only, doi=doi)
        else:
            # 标准模式
            extractor = TableExtractor(
                llm_client=self.multimodal_client,  # 使用multimodal_client
                prompt_path=self.table_prompt_path
            )
            # TableExtractor.extract expects (blocks, paper_image_dir, doi)
            records = extractor.extract(blocks_only, str(paper_dir), doi=doi)
        
        # 为每条记录添加block_id溯源
        if records and block_ids:
            records_per_block = max(1, len(records) // len(block_ids))
            for i, record in enumerate(records):
                block_index = min(i // records_per_block, len(block_ids) - 1)
                block_id = f"table_block_{block_ids[block_index]}"
                if "source_in_document" in record and isinstance(record["source_in_document"], dict):
                    record["source_in_document"]["block_id"] = block_id
                else:
                    record["block_id"] = block_id
        
        return records
    
    def _extract_from_figure(self, figure_blocks: List[Tuple[int, Dict]], doi: str, paper_dir: Path) -> List[Dict]:
        """
        从图像提取
        
        Args:
            figure_blocks: 图像块列表 [(block_id, block), ...]
            doi: DOI
            paper_dir: 论文目录（用于解析图像路径）
            
        Returns:
            提取的记录列表
        """
        # 提取block_ids和blocks
        block_ids = [bid for bid, _ in figure_blocks]
        blocks_only = [b for _, b in figure_blocks]
        
        # 添加图像路径 (support both 'image_path' and 'img_path')
        for block in blocks_only:
            # Handle both field names
            img_path = block.get('image_path') or block.get('img_path')
            if img_path:
                # Normalize to 'image_path' for FigureExtractor
                block['image_path'] = str(paper_dir / img_path)
        
        # 执行提取
        if self.use_multi_model_voting:
            # 多模型投票模式 - 图片只用单模型（GLM-4.6V），无需投票
            logger.info("  Using GLM-4.6V for figures (single model)...")
            
            all_records = []
            for block_id, block in figure_blocks:
                img_path = block.get('image_path') or block.get('img_path')
                if not img_path:
                    logger.warning(f"  ⚠️ Block {block_id} has no image path, skipping")
                    continue
                
                image_path = str(paper_dir / img_path)
                
                # 使用多模型提取器（实际只调用glm46v）
                records = self.multi_model_extractor.extract_from_figure(
                    image_path=image_path,
                    prompt_template=self._load_prompt_template(self.figure_prompt_path),
                    doi=doi,
                    block_id=block_id
                )
                
                all_records.extend(records)
            
            logger.info(f"  Extracted {len(all_records)} records from figures")
            records = all_records
            
        elif self.use_multi_agent:
            # Multi-Agent模式
            extractor = MultiAgentExtractor(
                llm_client=self.multimodal_client,  # 使用multimodal_client
                prompt_paths={
                    "text": self.text_prompt_path,
                    "table": self.table_prompt_path,
                    "figure": self.figure_prompt_path
                },
                source_type="figure"
            )
            records = extractor.extract(blocks_only, doi=doi)
        else:
            # 标准模式
            extractor = FigureExtractor(
                llm_client=self.multimodal_client,  # 使用multimodal_client
                prompt_path=self.figure_prompt_path
            )
            # FigureExtractor.extract expects (blocks, paper_image_dir, doi)
            records = extractor.extract(blocks_only, str(paper_dir), doi=doi)
        
        # 为每条记录添加block_id溯源
        if records and block_ids:
            records_per_block = max(1, len(records) // len(block_ids))
            for i, record in enumerate(records):
                block_index = min(i // records_per_block, len(block_ids) - 1)
                block_id = f"figure_block_{block_ids[block_index]}"
                if "source_in_document" in record and isinstance(record["source_in_document"], dict):
                    record["source_in_document"]["block_id"] = block_id
                else:
                    record["block_id"] = block_id
        
        return records
    
    def _clean_duplicate_fields(self, records: List[Dict]) -> List[Dict]:
        """
        清理重复字段（不同模型可能使用不同字段名）
        
        统一字段名，删除重复字段：
        - temperature vs temperature_value -> temperature
        - pH vs ph -> pH  
        - optimal_temperature vs optimal_temperature_value -> optimal_temperature
        - 删除其他非标准字段
        
        Args:
            records: 记录列表
            
        Returns:
            清理后的记录列表
        """
        # 标准字段列表（来自 aggregation prompt）
        STANDARD_FIELDS = {
            "enzyme_name", "enzyme_full_name", "enzyme_type", "ec_number", "gene_name",
            "uniprot_id", "genbank_id", "pdb_id", "sequence",
            "organism", "strain", "is_recombinant", "is_wild_type", "mutations",
            "substrate", "substrate_smiles", "substrate_concentration",
            "Km_value", "Km_unit", "Vmax_value", "Vmax_unit", "kcat_value", "kcat_unit",
            "kcat_Km_value", "kcat_Km_unit",
            "degradation_efficiency", "reaction_time_value", "reaction_time_unit",
            "products",
            "temperature_value", "temperature_unit", "ph", "optimal_ph", "optimal_temperature_value", "optimal_temperature_unit",
            "thermal_stability", "thermal_stability_unit", "thermal_stability_time", "thermal_stability_time_unit",
            "notes",
            "_aggregation_notes", "_model_comparison", "_confidence", "_source_location",
            # 保留的内部字段
            "_id", "_source_file", "_source_model", "_source_block_id", "_source_type",
            "source_in_document", "block_id", "confidence_score"
        }
        
        for record in records:
            # 字段名统一：保留 _value 形式的标准字段，删除旧的简化命名
            
            # temperature vs temperature_value: 保留 temperature_value
            if "temperature" in record and "temperature_value" not in record:
                # 如果只有 temperature，重命名为 temperature_value
                record["temperature_value"] = record["temperature"]
            # 删除旧字段
            if "temperature" in record:
                del record["temperature"]
            
            # pH vs ph: 保留 ph
            if "pH" in record and "ph" not in record:
                # 如果只有 pH，重命名为 ph
                record["ph"] = record["pH"]
            # 删除旧字段
            if "pH" in record:
                del record["pH"]
            
            # optimal_temperature vs optimal_temperature_value: 保留 optimal_temperature_value
            if "optimal_temperature" in record and "optimal_temperature_value" not in record:
                # 如果只有 optimal_temperature，重命名为 optimal_temperature_value
                record["optimal_temperature_value"] = record["optimal_temperature"]
            # 删除旧字段
            if "optimal_temperature" in record:
                del record["optimal_temperature"]
            
            # 删除其他非标准字段
            extra_fields = set(record.keys()) - STANDARD_FIELDS
            for field in extra_fields:
                del record[field]
        
        return records
    
    def _merge_enzyme_info_within_paper(self, records: List[Dict]) -> List[Dict]:
        """
        合并同一论文中相同酶的信息
        
        在同一篇论文内，相同enzyme_name的记录共享完整信息。
        
        策略：
        1. 基本信息字段（enzyme_full_name, organism等）在所有相同enzyme_name的记录间共享
        2. 条件特定字段（optimal_ph, optimal_temperature等）只在完全相同的酶变体间共享
           - 野生型和突变体的最优条件通常不同，需要分开处理
        3. 优先保留 confidence_score 高的记录的信息
        4. 优先保留非空值
        
        Args:
            records: 记录列表
            
        Returns:
            合并后的记录列表
        """
        from collections import defaultdict
        
        if not records:
            return records
        
        # 基本信息字段：可以在所有相同enzyme_name的记录间共享
        BASIC_INFO_FIELDS = [
            'enzyme_full_name',
            'enzyme_type',
            'ec_number',
            'gene_name',
        ]
        
        # 来源信息字段：相同organism/strain的记录间共享
        SOURCE_INFO_FIELDS = [
            'organism',
            'strain',
        ]
        
        # 条件特定字段：只在完全相同的酶变体（包括突变）间共享
        # 野生型和突变体的最优条件通常不同
        CONDITION_FIELDS = [
            'optimal_ph',
            'optimal_temperature_value',
            'optimal_temperature_unit',
        ]
        
        # 按 enzyme_name 分组
        enzyme_groups = defaultdict(list)
        for record in records:
            enzyme_name = record.get('enzyme_name')
            if enzyme_name:
                enzyme_groups[enzyme_name].append(record)
        
        filled_count = 0
        
        # 为每个酶构建信息模板
        for enzyme_name, group_records in enzyme_groups.items():
            if len(group_records) <= 1:
                continue  # 只有一条记录，无需合并
            
            # 按置信度和来源排序
            sorted_records = sorted(
                group_records,
                key=lambda r: (
                    r.get('confidence_score', 0),
                    'table_block_4' in r.get('source_in_document', {}).get('block_id', ''),
                    'text_block' in r.get('source_in_document', {}).get('block_id', ''),
                ),
                reverse=True
            )
            
            # 1. 构建基本信息模板（所有记录共享）
            basic_template = {}
            for field in BASIC_INFO_FIELDS:
                for record in sorted_records:
                    value = record.get(field)
                    if value not in (None, '', [], {}):
                        basic_template[field] = value
                        break
            
            # 2. 按 organism/strain 分组，构建来源信息模板
            source_groups = defaultdict(list)
            for record in group_records:
                organism = record.get('organism', 'Unknown')
                strain = record.get('strain', 'Unknown')
                source_key = f"{organism}|{strain}"
                source_groups[source_key].append(record)
            
            source_templates = {}
            for source_key, source_records in source_groups.items():
                source_template = {}
                sorted_source_records = sorted(
                    source_records,
                    key=lambda r: r.get('confidence_score', 0),
                    reverse=True
                )
                for field in SOURCE_INFO_FIELDS:
                    for record in sorted_source_records:
                        value = record.get(field)
                        if value not in (None, '', [], {}):
                            source_template[field] = value
                            break
                source_templates[source_key] = source_template
            
            # 3. 按变体类型（wild-type/mutations）分组，构建条件模板
            variant_groups = defaultdict(list)
            for record in group_records:
                is_wild = record.get('is_wild_type')
                mutations = record.get('mutations') or ''
                organism = record.get('organism', 'Unknown')
                # 使用 organism + wild_type/mutations 作为key
                if is_wild:
                    variant_key = f"{organism}|wild_type"
                else:
                    variant_key = f"{organism}|{mutations}"
                variant_groups[variant_key].append(record)
            
            condition_templates = {}
            for variant_key, variant_records in variant_groups.items():
                condition_template = {}
                sorted_variant_records = sorted(
                    variant_records,
                    key=lambda r: r.get('confidence_score', 0),
                    reverse=True
                )
                for field in CONDITION_FIELDS:
                    for record in sorted_variant_records:
                        value = record.get(field)
                        if value not in (None, '', [], {}):
                            condition_template[field] = value
                            break
                condition_templates[variant_key] = condition_template
            
            # 4. 补全每条记录
            for record in group_records:
                # 补全基本信息（所有记录）
                for field, value in basic_template.items():
                    if record.get(field) in (None, '', [], {}):
                        record[field] = value
                        filled_count += 1
                
                # 补全来源信息（相同organism/strain）
                organism = record.get('organism', 'Unknown')
                strain = record.get('strain', 'Unknown')
                source_key = f"{organism}|{strain}"
                if source_key in source_templates:
                    for field, value in source_templates[source_key].items():
                        if record.get(field) in (None, '', [], {}):
                            record[field] = value
                            filled_count += 1
                
                # 补全条件信息（相同变体）
                is_wild = record.get('is_wild_type')
                mutations = record.get('mutations') or ''
                if is_wild:
                    variant_key = f"{organism}|wild_type"
                else:
                    variant_key = f"{organism}|{mutations}"
                
                if variant_key in condition_templates:
                    for field, value in condition_templates[variant_key].items():
                        if record.get(field) in (None, '', [], {}):
                            record[field] = value
                            filled_count += 1
        
        if filled_count > 0:
            logger.info(f"    Filled {filled_count} missing fields across {len(enzyme_groups)} enzyme groups")
        
        return records
    
    def _run_multiagent_review(self, records: List[Dict], doi: str, content_list: Optional[List[Dict]] = None) -> Tuple[List[Dict], Optional[Dict]]:
        """
        运行 ReviewerAgent 智能审核
        
        使用 review_pipeline.py 中的 ReviewerAgent (LLM + 联网工具)
        
        Args:
            records: 待审核的记录列表
            doi: 论文DOI
            content_list: 论文的content_list（用于根据block_id提取相关段落）
            
        Returns:
            (审核后的记录列表, 审核摘要)
        """
        import asyncio
        from src.agents.review_pipeline import ReviewerAgent
        
        try:
            # 创建审核 Agent (使用review_client作为LLM)
            reviewer = ReviewerAgent(
                llm_client=self.review_client,  # Use review_client instead of text_client
                config={
                    "enable_web_verification": self.multiagent_enable_web,
                    "auto_approve_threshold": 0.85
                }
            )
            
            # 运行异步审核
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                review_result = loop.run_until_complete(
                    reviewer.review(records=records, doi=doi, content_list=content_list)
                )
            finally:
                loop.close()
            
            # 提取审核后的记录和摘要
            reviewed_records = review_result.records
            summary = {
                "decision": review_result.decision.value,
                "status_breakdown": {
                    "approved": review_result.summary.get("approved", 0),
                    "needs_review": review_result.summary.get("needs_review", 0),
                    "rejected": review_result.summary.get("rejected", 0)
                },
                "reasoning": review_result.reasoning
            }
            
            return reviewed_records, summary
            
        except Exception as e:
            logger.error(f"ReviewerAgent review failed: {e}")
            logger.debug(traceback.format_exc())
            return records, None
    
    def _save_paper_result(self, paper_name: str, records: List[Dict]):
        """保存单篇论文的中间结果"""
        output_file = self.output_dir / f"{paper_name}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save intermediate result for {paper_name}: {e}")
    
    def _save_final_results(self, all_results: Dict, failed_papers: Dict, output_dir: str):
        """
        保存最终结果并生成质量分析报告

        采用统计分析策略替代HITL逐篇审核：
        0. 过滤非霉菌毒素底物的记录
        1. 为每条记录计算confidence_score
        2. 保存所有提取结果
        3. 生成质量分析报告
        4. 抽样低质量案例用于prompt优化
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 霉菌毒素底物列表（用于快速匹配）
        MYCOTOXIN_SUBSTRATES = {
            # 黄曲霉毒素类
            'aflatoxin b1', 'aflatoxin b2', 'aflatoxin g1', 'aflatoxin g2', 'afb1', 'afb2', 'afg1', 'afg2',
            'aflatoxin m1', 'aflatoxin m2', 'afm1', 'afm2',
            # 赭曲霉毒素类
            'ochratoxin a', 'ochratoxin b', 'ota', 'otb',
            # 单端孢霉烯族
            'deoxynivalenol', 'don', 'nivalenol', 'niv',
            't-2 toxin', 't-2', 'ht-2 toxin', 'ht-2',
            '3-acetyldeoxynivalenol', '3-adon', '15-acetyldeoxynivalenol', '15-adon',
            'diacetoxyscirpenol', 'das',
            # 伏马毒素类
            'fumonisin b1', 'fumonisin b2', 'fumonisin b3', 'fb1', 'fb2', 'fb3',
            'hydrolyzed fumonisin b1', 'hfb1',
            # 玉米赤霉烯酮类
            'zearalenone', 'zen', 'zea',
            'alpha-zearalenol', 'α-zearalenol', 'α-zel', 'alpha-zel',
            'beta-zearalenol', 'β-zearalenol', 'β-zel', 'beta-zel',
            # 其他
            'patulin', 'citrinin', 'sterigmatocystin',
            'cyclopiazonic acid', 'cpa',
            'alternariol', 'aoh', 'alternariol monomethyl ether', 'ame',
            'tenuazonic acid', 'tea',
            'moniliformin', 'beauvericin', 'enniatin',
        }

        # 收集需要用LLM判断的未知底物
        unknown_substances = {}  # {substrate: [(paper_name, record), ...]}

        def is_known_mycotoxin(substrate: str) -> bool:
            """快速检查是否为已知霉菌毒素"""
            if not substrate:
                return False
            substrate_lower = substrate.lower().strip()
            for mycotoxin in MYCOTOXIN_SUBSTRATES:
                if mycotoxin in substrate_lower or substrate_lower in mycotoxin:
                    return True
            return False

        # 0. 两步过滤：先快速匹配已知毒素，再用LLM判断未知底物
        logger.info("🔍 Filtering non-mycotoxin substrate records...")

        # 第一步：分类 - 已知霉菌毒素 vs 未知底物
        known_records = {}  # {paper_name: [records]}
        for paper_name in all_results.keys():
            known_records[paper_name] = []
            for record in all_results[paper_name]:
                substrate = record.get('substrate', '')
                if is_known_mycotoxin(substrate):
                    known_records[paper_name].append(record)
                else:
                    # 收集未知底物，等待LLM判断
                    if substrate not in unknown_substances:
                        unknown_substances[substrate] = []
                    unknown_substances[substrate].append((paper_name, record))

        known_count = sum(len(r) for r in known_records.values())
        unknown_count = sum(len(recs) for recs in unknown_substances.values())
        logger.info(f"   ✓ Known mycotoxins: {known_count} records")
        logger.info(f"   ? Unknown substrates: {unknown_count} records ({len(unknown_substances)} unique substrates)")

        # 第二步：用DeepSeek判断未知底物
        if unknown_substances:
            logger.info(f"   🤖 Using DeepSeek to classify {len(unknown_substances)} unknown substrates...")

            # 构建判断prompt
            judge_prompt = "You are a mycotoxin expert. Determine if each substrate is a mycotoxin or related compound.\n\n"
            judge_prompt += "## Mycotoxin Categories:\n"
            judge_prompt += "- Aflatoxins (AFB1, AFB2, AFG1, AFG2, AFM1, etc.)\n"
            judge_prompt += "- Ochratoxins (OTA, OTB)\n"
            judge_prompt += "- Trichothecenes (DON, NIV, T-2, HT-2, etc.)\n"
            judge_prompt += "- Fumonisins (FB1, FB2, FB3)\n"
            judge_prompt += "- Zearalenone and derivatives (ZEN, α-ZEL, β-ZEL)\n"
            judge_prompt += "- Others (Patulin, Citrinin, Sterigmatocystin, etc.)\n\n"
            judge_prompt += "## Common NON-mycotoxin substrates (REJECT):\n"
            judge_prompt += "- Enzyme assay substrates: ABTS, guaiacol, pNPP, catechol, DMP\n"
            judge_prompt += "- Dyes: RBBR, RB5, methylene blue\n"
            judge_prompt += "- General compounds: glucose, starch, cellulose\n\n"
            judge_prompt += "## Substrates to classify:\n\n"

            for i, substrate in enumerate(unknown_substances.keys(), 1):
                judge_prompt += f"{i}. {substrate}\n"

            judge_prompt += "\n## Output Format (JSON only):\n"
            judge_prompt += "```json\n"
            judge_prompt += "{\n"
            judge_prompt += '  "results": [\n'
            judge_prompt += '    {"substrate": "substrate name", "is_mycotoxin": true/false, "reason": "brief reason"},\n'
            judge_prompt += "    ...\n"
            judge_prompt += "  ]\n"
            judge_prompt += "}\n"
            judge_prompt += "```\n"

            try:
                response = self.text_client.generate(judge_prompt)

                # 解析响应
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                else:
                    response_text = response

                judgment = json.loads(response_text)

                # 根据判断结果分类记录
                mycotoxin_substances = set()
                for result in judgment.get('results', []):
                    if result.get('is_mycotoxin', False):
                        mycotoxin_substances.add(result['substrate'])
                        logger.debug(f"   ✓ {result['substrate']}: mycotoxin ({result.get('reason', '')})")
                    else:
                        logger.debug(f"   ✗ {result['substrate']}: NOT mycotoxin ({result.get('reason', '')})")

                # 将LLM判断为霉菌毒素的记录添加到结果中
                for substrate, records in unknown_substances.items():
                    if substrate in mycotoxin_substances:
                        for paper_name, record in records:
                            known_records[paper_name].append(record)

                llm_passed = sum(len(recs) for recs in unknown_substances.values()) - sum(
                    len([r for s, recs in unknown_substances.items() if s not in mycotoxin_substances for r in recs])
                )
                llm_filtered = unknown_count - llm_passed
                logger.info(f"   ✓ LLM classification: {llm_passed} passed, {llm_filtered} filtered")

            except Exception as e:
                logger.warning(f"   ⚠️ LLM classification failed: {e}")
                logger.warning(f"   ⚠️ Filtering ALL unknown substrates (conservative approach)")
                # 失败时保守策略：过滤掉所有未知底物
                # 这样可以确保只有明确的霉菌毒素被保留
                pass  # 不添加任何未知底物到known_records

        # 更新all_results
        all_results = known_records

        total_before = known_count + unknown_count
        total_after = sum(len(r) for r in all_results.values())
        logger.info(f"   ✓ Final: {total_before} → {total_after} records ({total_before - total_after} filtered)")

        # 1. 计算每条记录的confidence_score（3级评分系统）
        logger.info("📊 Calculating confidence scores (3-level system)...")
        for paper_name, records in all_results.items():
            for record in records:
                score = DataValidator.calculate_confidence(record)
                record["confidence_score"] = score
        logger.info(f"   ✓ Calculated scores for {sum(len(r) for r in all_results.values())} records")

        # 1. 保存完整提取结果
        results_file = Path(output_dir) / f"extraction_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 All extraction results saved to: {results_file}")

        # 2. 生成质量分析报告（替代HITL）
        quality_report = None  # 初始化，防止异常时变量未定义
        try:
            analyzer = QualityAnalyzer(sample_size=20)

            # 质量分析
            quality_report = analyzer.analyze_quality(all_results)

            # 抽取低质量样本
            low_quality_samples = analyzer.sample_low_quality(all_results, min_score=2)

            # 保存报告
            report_files = analyzer.save_reports(quality_report, low_quality_samples, output_dir)

            logger.info(f"📊 Quality analysis report saved to: {report_files['quality_report']}")
            logger.info(f"📝 Low-quality samples saved to: {report_files['samples']}")
            logger.info(f"💡 Prompt optimization guide saved to: {report_files['optimization_guide']}")

            # 打印质量摘要
            logger.info("")
            logger.info("=" * 60)
            logger.info("📊 QUALITY ANALYSIS SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total records: {quality_report['total_records']}")
            logger.info("Quality distribution:")
            for level, count in quality_report['quality_distribution'].items():
                pct = quality_report['quality_percentages'].get(level.split()[0], 0)
                logger.info(f"  - {level}: {count} ({pct:.1f}%)")
            logger.info(f"")
            logger.info(f"💡 Check {report_files['optimization_guide']} for prompt optimization suggestions")
            logger.info("=" * 60)

        except Exception as e:
            logger.warning(f"⚠️ Quality analysis failed: {e}")

        # 3. 保存流水线统计信息
        stats_file = Path(output_dir) / f"pipeline_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "statistics": self.stats.to_dict(),
                "failed_papers": failed_papers,
                "quality_report": quality_report if 'quality_report' in locals() else None
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Pipeline statistics saved to: {stats_file}")
    
    def _print_summary(self):
        """打印处理总结"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total papers: {self.stats.total_papers}")
        logger.info(f"✅ Processed: {self.stats.processed_papers}")
        logger.info(f"❌ Failed: {self.stats.failed_papers}")
        success_rate = (self.stats.processed_papers / self.stats.total_papers * 100 
                       if self.stats.total_papers > 0 else 0)
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("")
        logger.info(f"Total records extracted: {self.stats.total_records}")
        logger.info(f"  - Text: {self.stats.text_records}")
        logger.info(f"  - Table: {self.stats.table_records}")
        logger.info(f"  - Figure: {self.stats.figure_records}")
        
        # 合并统计
        if self.stats.records_merged > 0:
            logger.info("")
            logger.info("🔄 RECORD MERGE (Entity Alignment)")
            logger.info(f"  - Before merge: {self.stats.records_before_merge}")
            logger.info(f"  - After merge:  {self.stats.records_after_merge}")
            logger.info(f"  - Duplicates removed: {self.stats.records_merged} "
                       f"({self.stats.records_merged / self.stats.records_before_merge * 100:.1f}%)")
        
        logger.info("")
        logger.info(f"⏱️  Total time: {self.stats.total_time:.1f}s ({self.stats.total_time / 60:.1f}m)")
        logger.info(f"⏱️  Avg time per paper: {self.stats.avg_time_per_paper:.1f}s")
        if self.stats.total_papers > 1:
            sequential_time = self.stats.avg_time_per_paper * self.stats.total_papers
            speedup = sequential_time / self.stats.total_time if self.stats.total_time > 0 else 0
            logger.info(f"🚀 Speedup: {speedup:.2f}x (vs sequential)")
        
        # Token usage statistics
        logger.info("")
        logger.info("💰 TOKEN USAGE STATISTICS")
        logger.info("-" * 40)
        try:
            from src.llm_clients.providers import ZhipuAIClient
            token_stats = ZhipuAIClient.get_global_token_stats()
            logger.info(f"  Prompt tokens:      {token_stats['prompt_tokens']:,}")
            logger.info(f"  Completion tokens:  {token_stats['completion_tokens']:,}")
            logger.info(f"  Total tokens:       {token_stats['total_tokens']:,}")
            logger.info(f"  API requests:       {token_stats['request_count']}")
            
            # Cost estimation (GLM-4-Flash pricing: input ¥0.4/M, output ¥0.8/M)
            input_cost = token_stats['prompt_tokens'] / 1_000_000 * 0.4
            output_cost = token_stats['completion_tokens'] / 1_000_000 * 0.8
            total_cost = input_cost + output_cost
            logger.info(f"")
            logger.info(f"  💵 Estimated cost:  ¥{total_cost:.4f}")
            logger.info(f"     Input:  ¥{input_cost:.4f} ({token_stats['prompt_tokens']:,} tokens @ ¥0.4/M)")
            logger.info(f"     Output: ¥{output_cost:.4f} ({token_stats['completion_tokens']:,} tokens @ ¥0.8/M)")
        except Exception as e:
            logger.warning(f"Could not retrieve token statistics: {e}")
        
        logger.info("=" * 80)
