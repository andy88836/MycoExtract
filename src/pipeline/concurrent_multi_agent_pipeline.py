"""
并发 Multi-Agent 提取流水线
支持多线程/多进程并发处理大量论文，大幅提升处理效率
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass, asdict
import traceback

from src.llm_extraction.multi_agent_extractor import MultiAgentExtractor
from src.llm_clients.providers import ZhipuAIClient

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_papers: int = 0
    processed_papers: int = 0
    failed_papers: int = 0
    total_records: int = 0
    text_records: int = 0
    table_records: int = 0
    figure_records: int = 0
    total_time: float = 0.0
    avg_time_per_paper: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ConcurrentMultiAgentPipeline:
    """
    并发 Multi-Agent 提取流水线
    
    特性:
    1. 支持多线程并发处理 (适合 I/O 密集型，推荐)
    2. 支持多进程并发处理 (适合 CPU 密集型)
    3. 实时进度跟踪
    4. 异常隔离：单个论文失败不影响其他
    5. 自动重试机制
    6. 结果实时保存
    """
    
    def __init__(
        self,
        llm_client: ZhipuAIClient,
        prompt_paths: Dict[str, str],
        max_workers: int = 5,
        use_multiprocessing: bool = False,
        max_retries: int = 2,
        save_intermediate: bool = True,
        output_dir: str = "results"
    ):
        """
        初始化并发流水线
        
        Args:
            llm_client: LLM 客户端
            prompt_paths: prompt 路径字典 {"text": path, "table": path, "figure": path}
            max_workers: 最大并发数 (推荐 3-10，取决于 API 限制)
            use_multiprocessing: 是否使用多进程 (默认 False，使用多线程)
            max_retries: 失败重试次数
            save_intermediate: 是否实时保存中间结果
            output_dir: 输出目录
        """
        self.llm_client = llm_client
        self.prompt_paths = prompt_paths
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.max_retries = max_retries
        self.save_intermediate = save_intermediate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程安全的锁
        self.lock = Lock()
        
        # 统计信息
        self.stats = ProcessingStats()
        
        logger.info(f"✓ Initialized ConcurrentMultiAgentPipeline")
        logger.info(f"  - Max workers: {max_workers}")
        logger.info(f"  - Mode: {'Multiprocessing' if use_multiprocessing else 'Multithreading'}")
        logger.info(f"  - Max retries: {max_retries}")
        logger.info(f"  - Output dir: {output_dir}")
    
    def run(
        self,
        paper_dirs: List[Path],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        并发处理多篇论文
        
        Args:
            paper_dirs: 论文目录列表
            progress_callback: 进度回调函数 callback(current, total, paper_name)
        
        Returns:
            {
                "results": {paper_name: records_list},
                "statistics": ProcessingStats,
                "failed_papers": {paper_name: error_message}
            }
        """
        start_time = time.time()
        self.stats.total_papers = len(paper_dirs)
        
        logger.info("=" * 80)
        logger.info(f"🚀 Starting concurrent processing: {len(paper_dirs)} papers")
        logger.info(f"   Workers: {self.max_workers}")
        logger.info("=" * 80)
        
        all_results = {}
        failed_papers = {}
        
        # 选择执行器
        ExecutorClass = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self._process_single_paper, paper_dir): paper_dir
                for paper_dir in paper_dirs
            }
            
            # 收集结果
            for future in as_completed(future_to_paper):
                paper_dir = future_to_paper[future]
                paper_name = paper_dir.name
                
                try:
                    # 获取结果
                    result = future.result()
                    
                    with self.lock:
                        if result['success']:
                            all_results[paper_name] = result['records']
                            self.stats.processed_papers += 1
                            self.stats.total_records += result['stats']['total']
                            self.stats.text_records += result['stats']['text']
                            self.stats.table_records += result['stats']['table']
                            self.stats.figure_records += result['stats']['figure']
                            
                            logger.info(f"✅ [{self.stats.processed_papers}/{self.stats.total_papers}] "
                                      f"{paper_name}: {result['stats']['total']} records "
                                      f"(Text: {result['stats']['text']}, "
                                      f"Table: {result['stats']['table']}, "
                                      f"Figure: {result['stats']['figure']})")
                        else:
                            failed_papers[paper_name] = result['error']
                            self.stats.failed_papers += 1
                            logger.error(f"❌ [{self.stats.processed_papers + self.stats.failed_papers}/"
                                       f"{self.stats.total_papers}] {paper_name}: {result['error']}")
                        
                        # 进度回调
                        if progress_callback:
                            progress_callback(
                                self.stats.processed_papers + self.stats.failed_papers,
                                self.stats.total_papers,
                                paper_name
                            )
                        
                        # 实时保存中间结果
                        if self.save_intermediate and result['success']:
                            self._save_paper_result(paper_name, result['records'])
                
                except Exception as e:
                    with self.lock:
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
        self._save_final_results(all_results, failed_papers)
        
        # 打印总结
        self._print_summary()
        
        return {
            "results": all_results,
            "statistics": self.stats.to_dict(),
            "failed_papers": failed_papers
        }
    
    def _process_single_paper(self, paper_dir: Path) -> Dict[str, Any]:
        """
        处理单篇论文（带重试机制）
        
        Args:
            paper_dir: 论文目录
        
        Returns:
            {
                "success": bool,
                "records": List[Dict],
                "stats": {"total": int, "text": int, "table": int, "figure": int},
                "error": str (if success=False)
            }
        """
        paper_name = paper_dir.name
        
        for attempt in range(self.max_retries + 1):
            try:
                # 读取 content_list.json
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
                
                # 分类 blocks
                text_blocks = [b for b in content_list if b['type'] == 'text']
                table_blocks = [b for b in content_list if b['type'] == 'table']
                figure_blocks = [b for b in content_list if b['type'] == 'figure']
                
                all_records = []
                
                # 提取文本
                if text_blocks:
                    text_extractor = MultiAgentExtractor(
                        llm_client=self.llm_client,
                        prompt_paths=self.prompt_paths,
                        source_type="text"
                    )
                    text_records = text_extractor.extract(text_blocks, doi=paper_name)
                    all_records.extend(text_records)
                
                # 提取表格
                if table_blocks:
                    table_extractor = MultiAgentExtractor(
                        llm_client=self.llm_client,
                        prompt_paths=self.prompt_paths,
                        source_type="table"
                    )
                    table_records = table_extractor.extract(table_blocks, doi=paper_name)
                    all_records.extend(table_records)
                
                # 提取图像
                if figure_blocks:
                    # 添加图像路径信息
                    for block in figure_blocks:
                        if 'image_path' in block:
                            # 转换为绝对路径
                            block['image_path'] = str(paper_dir / block['image_path'])
                    
                    figure_extractor = MultiAgentExtractor(
                        llm_client=self.llm_client,
                        prompt_paths=self.prompt_paths,
                        source_type="figure"
                    )
                    figure_records = figure_extractor.extract(figure_blocks, doi=paper_name)
                    all_records.extend(figure_records)
                
                # 统计
                stats = {
                    "total": len(all_records),
                    "text": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'text']),
                    "table": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'table']),
                    "figure": len([r for r in all_records if r.get('source_in_document', {}).get('source_type') == 'figure'])
                }
                
                return {
                    "success": True,
                    "records": all_records,
                    "stats": stats,
                    "error": None
                }
            
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"⚠️ {paper_name}: Attempt {attempt + 1} failed, retrying... ({str(e)})")
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
    
    def _save_paper_result(self, paper_name: str, records: List[Dict]):
        """保存单篇论文的中间结果"""
        output_file = self.output_dir / f"{paper_name}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save intermediate result for {paper_name}: {e}")
    
    def _save_final_results(self, all_results: Dict, failed_papers: Dict):
        """保存最终结果"""
        # 保存所有提取结果
        results_file = self.output_dir / "all_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "statistics": self.stats.to_dict(),
                "failed_papers": failed_papers
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Results saved to: {results_file}")
        logger.info(f"💾 Statistics saved to: {stats_file}")
    
    def _print_summary(self):
        """打印处理总结"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total papers: {self.stats.total_papers}")
        logger.info(f"✅ Processed: {self.stats.processed_papers}")
        logger.info(f"❌ Failed: {self.stats.failed_papers}")
        logger.info(f"Success rate: {self.stats.processed_papers / self.stats.total_papers * 100:.1f}%")
        logger.info("")
        logger.info(f"Total records extracted: {self.stats.total_records}")
        logger.info(f"  - Text: {self.stats.text_records}")
        logger.info(f"  - Table: {self.stats.table_records}")
        logger.info(f"  - Figure: {self.stats.figure_records}")
        logger.info("")
        logger.info(f"⏱️  Total time: {self.stats.total_time:.1f}s ({self.stats.total_time / 60:.1f}m)")
        logger.info(f"⏱️  Avg time per paper: {self.stats.avg_time_per_paper:.1f}s")
        if self.stats.total_papers > 1:
            sequential_time = self.stats.avg_time_per_paper * self.stats.total_papers
            speedup = sequential_time / self.stats.total_time if self.stats.total_time > 0 else 0
            logger.info(f"🚀 Speedup: {speedup:.2f}x (estimated)")
        logger.info("=" * 80)


class BatchProcessor:
    """
    批处理工具类
    支持将大量论文分批处理，避免内存溢出
    """
    
    def __init__(
        self,
        pipeline: ConcurrentMultiAgentPipeline,
        batch_size: int = 50
    ):
        """
        初始化批处理器
        
        Args:
            pipeline: 并发流水线
            batch_size: 每批处理的论文数量
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
        logger.info(f"✓ BatchProcessor initialized with batch_size={batch_size}")
    
    def process_in_batches(
        self,
        paper_dirs: List[Path],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        分批处理论文
        
        Args:
            paper_dirs: 论文目录列表
            progress_callback: 进度回调
        
        Returns:
            合并后的所有批次结果
        """
        total_papers = len(paper_dirs)
        num_batches = (total_papers + self.batch_size - 1) // self.batch_size
        
        logger.info(f"📦 Processing {total_papers} papers in {num_batches} batches")
        logger.info(f"   Batch size: {self.batch_size}")
        
        all_results = {}
        all_failed = {}
        combined_stats = ProcessingStats()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, total_papers)
            batch = paper_dirs[start_idx:end_idx]
            
            logger.info("")
            logger.info(f"{'=' * 80}")
            logger.info(f"📦 BATCH {batch_idx + 1}/{num_batches}")
            logger.info(f"   Papers: {start_idx + 1}-{end_idx} of {total_papers}")
            logger.info(f"{'=' * 80}")
            
            # 处理当前批次
            result = self.pipeline.run(batch, progress_callback)
            
            # 合并结果
            all_results.update(result['results'])
            all_failed.update(result['failed_papers'])
            
            # 累加统计
            stats = result['statistics']
            combined_stats.total_papers += stats['total_papers']
            combined_stats.processed_papers += stats['processed_papers']
            combined_stats.failed_papers += stats['failed_papers']
            combined_stats.total_records += stats['total_records']
            combined_stats.text_records += stats['text_records']
            combined_stats.table_records += stats['table_records']
            combined_stats.figure_records += stats['figure_records']
            combined_stats.total_time += stats['total_time']
        
        # 计算总体平均时间
        if combined_stats.processed_papers > 0:
            combined_stats.avg_time_per_paper = (
                combined_stats.total_time / combined_stats.processed_papers
            )
        
        # 保存合并结果
        self._save_combined_results(all_results, all_failed, combined_stats)
        
        return {
            "results": all_results,
            "statistics": combined_stats.to_dict(),
            "failed_papers": all_failed
        }
    
    def _save_combined_results(
        self,
        all_results: Dict,
        all_failed: Dict,
        stats: ProcessingStats
    ):
        """保存合并后的结果"""
        output_dir = self.pipeline.output_dir
        
        # 保存总结果
        combined_file = output_dir / "combined_all_results.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 保存总统计
        combined_stats_file = output_dir / "combined_statistics.json"
        with open(combined_stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "statistics": stats.to_dict(),
                "failed_papers": all_failed
            }, f, ensure_ascii=False, indent=2)
        
        logger.info("")
        logger.info(f"💾 Combined results saved to: {combined_file}")
        logger.info(f"💾 Combined statistics saved to: {combined_stats_file}")
