"""
PDF Direct Extraction Pipeline - End-to-End Multimodal Approach

直接将PDF页面图像传给多模态LLM (GLM-4.6V)，进行端到端提取。
与分块提取方式对比测试。

特点：
1. PDF → 图像转换 (每页一张图)
2. 所有页面一次性传给LLM
3. 单次API调用完成整篇论文的提取
4. 更简单的流程，但可能消耗更多token
"""

import os
import json
import logging
import time
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import base64

from src.llm_clients.providers import BaseLLMClient, ZhipuAIClient
from src.utils.data_validator import DataValidator

logger = logging.getLogger(__name__)


@dataclass
class PDFPipelineStats:
    """PDF直接提取流水线统计"""
    total_papers: int = 0
    processed_papers: int = 0
    failed_papers: int = 0
    total_records: int = 0
    total_pages: int = 0
    total_time: float = 0.0
    avg_time_per_paper: float = 0.0
    
    # Token统计
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PDFDirectPipeline:
    """
    PDF端到端提取流水线
    
    直接将PDF页面图像传给GLM-4.6V，一次性提取所有酶动力学数据。
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_path: str = "prompts/prompts_extract_from_pdf.txt",
        max_workers: int = 2,
        max_pages: int = 30,  # 最大处理页数
        dpi: int = 150,  # 图像分辨率
        max_retries: int = 2,
        save_intermediate: bool = True
    ):
        """
        Args:
            llm_client: 多模态LLM客户端 (GLM-4.6V)
            prompt_path: PDF提取提示词路径
            max_workers: 最大并发论文数
            max_pages: 每篇PDF最大处理页数
            dpi: PDF转图像的分辨率
            max_retries: 失败重试次数
            save_intermediate: 是否保存中间结果
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompt_path)
        self.max_workers = max_workers
        self.max_pages = max_pages
        self.dpi = dpi
        self.max_retries = max_retries
        self.save_intermediate = save_intermediate
        
        self._lock = Lock()
        self.stats = PDFPipelineStats()
        
        logger.info(f"✓ PDF Direct Pipeline initialized")
        logger.info(f"  - LLM: {type(llm_client).__name__}")
        logger.info(f"  - Max pages per PDF: {max_pages}")
        logger.info(f"  - DPI: {dpi}")
        logger.info(f"  - Max workers: {max_workers}")
    
    def _load_prompt(self, prompt_path: str) -> str:
        """加载提示词模板"""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
    
    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        将PDF转换为base64编码的图像列表
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            base64编码图像列表
        """
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = min(len(doc), self.max_pages)
            
            logger.info(f"  Converting PDF to images: {total_pages} pages")
            
            # 设置缩放矩阵
            zoom = self.dpi / 72  # 72是PDF的默认DPI
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat)
                
                # 转为PNG bytes
                img_bytes = pix.tobytes("png")
                
                # Base64编码
                b64_img = base64.b64encode(img_bytes).decode('utf-8')
                images.append(b64_img)
            
            doc.close()
            
            with self._lock:
                self.stats.total_pages += total_pages
            
            return images
            
        except Exception as e:
            logger.error(f"  Error converting PDF to images: {e}")
            raise
    
    def extract_from_pdf(self, pdf_path: str, doi: str = "unknown") -> List[Dict]:
        """
        从PDF直接提取数据
        
        Args:
            pdf_path: PDF文件路径
            doi: DOI
            
        Returns:
            提取的记录列表
        """
        # 1. 转换PDF为图像
        images = self.pdf_to_images(pdf_path)
        
        if not images:
            logger.warning(f"  No images extracted from PDF")
            return []
        
        logger.info(f"  Sending {len(images)} page images to LLM...")
        
        # 2. 构建消息
        content = []
        
        # 添加文本提示
        content.append({
            "type": "text",
            "text": self.prompt_template
        })
        
        # 添加所有页面图像
        for i, img_b64 in enumerate(images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })
        
        messages = [
            {"role": "user", "content": content}
        ]
        
        # 3. 调用LLM
        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=8192
            )
            
            # 记录token使用
            if hasattr(response, 'usage') and response.usage:
                with self._lock:
                    self.stats.prompt_tokens += response.usage.prompt_tokens or 0
                    self.stats.completion_tokens += response.usage.completion_tokens or 0
                    self.stats.total_tokens += (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
                    self.stats.api_requests += 1
                    
                logger.info(f"  Token usage - Prompt: {response.usage.prompt_tokens}, "
                           f"Completion: {response.usage.completion_tokens}")
            
            result_text = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"  LLM API call failed: {e}")
            raise
        
        # 4. 解析JSON结果
        records = self._parse_response(result_text, doi)
        
        return records
    
    def _parse_response(self, response_text: str, doi: str) -> List[Dict]:
        """解析LLM返回的JSON"""
        try:
            # 清理响应文本
            text = response_text.strip()
            
            # 移除可能的markdown代码块
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            text = text.strip()
            
            # 解析JSON
            records = json.loads(text)
            
            if not isinstance(records, list):
                records = [records]
            
            # 添加DOI
            for record in records:
                record['doi'] = doi
                record['extraction_method'] = 'pdf_direct'
                if 'source_in_document' not in record:
                    record['source_in_document'] = {
                        'source_type': 'pdf_direct',
                        'method': 'end-to-end multimodal'
                    }
            
            logger.info(f"  ✓ Extracted {len(records)} records")
            return records
            
        except json.JSONDecodeError as e:
            logger.error(f"  JSON parsing error: {e}")
            logger.debug(f"  Response text: {response_text[:500]}...")
            return []
    
    def _process_paper(self, paper_info: Dict) -> Dict[str, Any]:
        """
        处理单篇论文
        
        Args:
            paper_info: {"pdf_path": str, "doi": str, "name": str}
            
        Returns:
            处理结果
        """
        pdf_path = paper_info['pdf_path']
        doi = paper_info.get('doi', Path(pdf_path).stem)
        name = paper_info.get('name', Path(pdf_path).stem)
        
        for attempt in range(self.max_retries + 1):
            try:
                records = self.extract_from_pdf(pdf_path, doi)
                
                # 验证数据
                if records:
                    records = DataValidator.validate_batch(records)
                
                return {
                    "success": True,
                    "records": records,
                    "count": len(records),
                    "error": None
                }
                
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"  Attempt {attempt + 1} failed, retrying... ({str(e)})")
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "success": False,
                        "records": [],
                        "count": 0,
                        "error": str(e)
                    }
    
    def run(
        self,
        paper_list: List[Dict],
        output_dir: str = "results"
    ) -> Dict[str, Any]:
        """
        批量处理论文
        
        Args:
            paper_list: [{"pdf_path": str, "doi": str, "name": str}, ...]
            output_dir: 输出目录
            
        Returns:
            处理结果汇总
        """
        start_time = time.time()
        self.stats = PDFPipelineStats()  # 重置统计
        self.stats.total_papers = len(paper_list)
        
        # 重置ZhipuAI token计数器
        ZhipuAIClient.reset_global_stats()
        
        logger.info("=" * 80)
        logger.info("🚀 PDF Direct Pipeline Starting")
        logger.info(f"   Papers: {len(paper_list)}")
        logger.info(f"   Workers: {self.max_workers}")
        logger.info(f"   Max pages: {self.max_pages}")
        logger.info("=" * 80)
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = Path(output_dir)
        
        all_results = {}
        failed_papers = {}
        
        # 并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {
                executor.submit(self._process_paper, paper): paper
                for paper in paper_list
            }
            
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                paper_name = paper.get('name', Path(paper['pdf_path']).stem)
                
                try:
                    result = future.result()
                    
                    with self._lock:
                        if result['success']:
                            all_results[paper_name] = result['records']
                            self.stats.processed_papers += 1
                            self.stats.total_records += result['count']
                            
                            logger.info(
                                f"✅ [{self.stats.processed_papers}/{self.stats.total_papers}] "
                                f"{paper_name}: {result['count']} records"
                            )
                            
                            # 保存中间结果
                            if self.save_intermediate:
                                self._save_paper_result(paper_name, result['records'])
                        else:
                            failed_papers[paper_name] = result['error']
                            self.stats.failed_papers += 1
                            logger.error(
                                f"❌ [{self.stats.processed_papers + self.stats.failed_papers}/"
                                f"{self.stats.total_papers}] {paper_name}: {result['error']}"
                            )
                
                except Exception as e:
                    with self._lock:
                        failed_papers[paper_name] = str(e)
                        self.stats.failed_papers += 1
                        logger.error(f"❌ {paper_name}: {e}")
        
        # 更新统计
        self.stats.total_time = time.time() - start_time
        if self.stats.processed_papers > 0:
            self.stats.avg_time_per_paper = self.stats.total_time / self.stats.processed_papers
        
        # 获取全局token统计
        token_stats = ZhipuAIClient.get_global_token_stats()
        self.stats.prompt_tokens = token_stats['prompt_tokens']
        self.stats.completion_tokens = token_stats['completion_tokens']
        self.stats.total_tokens = token_stats['total_tokens']
        self.stats.api_requests = token_stats['request_count']
        
        # 保存最终结果
        self._save_final_results(all_results, failed_papers, output_dir)
        
        # 打印总结
        self._print_summary()
        
        return {
            "results": all_results,
            "statistics": self.stats.to_dict(),
            "failed_papers": failed_papers
        }
    
    def _save_paper_result(self, paper_name: str, records: List[Dict]):
        """保存单篇论文结果"""
        output_file = self.output_dir / f"{paper_name}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save result for {paper_name}: {e}")
    
    def _save_final_results(self, all_results: Dict, failed_papers: Dict, output_dir: str):
        """保存最终结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存所有结果
        results_file = Path(output_dir) / f"pdf_direct_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Results saved to: {results_file}")
        
        # 保存统计信息
        stats_file = Path(output_dir) / f"pdf_direct_statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "statistics": self.stats.to_dict(),
                "failed_papers": failed_papers
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Statistics saved to: {stats_file}")
    
    def _print_summary(self):
        """打印处理总结"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 PDF DIRECT PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total papers: {self.stats.total_papers}")
        logger.info(f"✅ Processed: {self.stats.processed_papers}")
        logger.info(f"❌ Failed: {self.stats.failed_papers}")
        success_rate = (self.stats.processed_papers / self.stats.total_papers * 100 
                       if self.stats.total_papers > 0 else 0)
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("")
        logger.info(f"Total records extracted: {self.stats.total_records}")
        logger.info(f"Total pages processed: {self.stats.total_pages}")
        logger.info("")
        logger.info(f"⏱️  Total time: {self.stats.total_time:.1f}s ({self.stats.total_time / 60:.1f}m)")
        logger.info(f"⏱️  Avg time per paper: {self.stats.avg_time_per_paper:.1f}s")
        logger.info("")
        logger.info("💰 TOKEN USAGE STATISTICS")
        logger.info("-" * 40)
        logger.info(f"  Prompt tokens:      {self.stats.prompt_tokens:,}")
        logger.info(f"  Completion tokens:  {self.stats.completion_tokens:,}")
        logger.info(f"  Total tokens:       {self.stats.total_tokens:,}")
        logger.info(f"  API requests:       {self.stats.api_requests}")
        
        # 成本估算 (GLM-4.6V 价格待确认，这里用估算值)
        # 假设: input ¥2/M, output ¥2/M (视觉模型通常更贵)
        input_cost = self.stats.prompt_tokens / 1_000_000 * 2.0
        output_cost = self.stats.completion_tokens / 1_000_000 * 2.0
        total_cost = input_cost + output_cost
        logger.info(f"")
        logger.info(f"  💵 Estimated cost:  ¥{total_cost:.4f}")
        logger.info(f"     (Based on ¥2/M input, ¥2/M output - check actual pricing)")
        logger.info("=" * 80)
