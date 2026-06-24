"""
Multi-Agent协作提取器 - 三Agent流水线模式

模拟人类团队工作方式：
- Agent A (Extractor): 高召回率，宽松提取
- Agent B (Reviewer): 高精确率，严格审核  
- Agent C (Synthesizer): 综合生成最终结果

Reference:
    "Multi-Agent Collaboration Improves Information Extraction"
    - Extractor-Reviewer-Synthesizer (ERS) Pipeline
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class MultiAgentExtractor:
    """
    Multi-Agent协作提取器
    
    三Agent流水线：
    1. Extractor: 提取初稿（高召回）
    2. Reviewer: 审核并指出错误（高精确）
    3. Synthesizer: 综合生成最终版本
    """
    
    def __init__(
        self,
        llm_client,
        source_type: str,
        prompt_paths: Dict[str, str],
        context_overlap_sentences: int = 2
    ):
        """
        Args:
            llm_client: LLM客户端
            source_type: 来源类型（text/table/figure）
            prompt_paths: 提示词路径字典
            context_overlap_sentences: 上下文重叠句子数
        """
        self.llm_client = llm_client
        self.source_type = source_type
        self.prompt_paths = prompt_paths
        self.context_overlap_sentences = context_overlap_sentences
        
        # 加载原始提示词
        self.extractor_prompt = self._load_prompt(prompt_paths[source_type])
        
        # 构建Agent角色提示词
        self.reviewer_prompt = self._build_reviewer_prompt()
        self.synthesizer_prompt = self._build_synthesizer_prompt()
        
        logger.info(f"✓ MultiAgentExtractor initialized for {source_type}")
        logger.info(f"  - Mode: Extractor-Reviewer-Synthesizer")
    
    def extract(
        self,
        blocks: List[Dict[str, Any]],
        doi: str,
        paper_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        使用三Agent流水线提取数据
        
        Args:
            blocks: 数据blocks
            doi: 论文DOI
            paper_dir: 论文目录（table/figure需要）
            
        Returns:
            提取的fragments列表
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🤝 Multi-Agent Extraction - {self.source_type.upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"Blocks: {len(blocks)}")
        
        all_fragments = []
        futures = []
        
        # 并行处理 blocks
        # Multi-Agent 比较耗时且 token 消耗大，控制并发数
        max_workers = 5
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, block in enumerate(blocks, 1):
                # 提交任务到线程池
                future = executor.submit(self._process_single_block, block, i, len(blocks), doi, paper_dir)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    fragments = future.result()
                    if fragments:
                        all_fragments.extend(fragments)
                except Exception as e:
                    import traceback
                    logger.error(f"Error in parallel multi-agent extraction: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Multi-Agent Extraction Complete: {len(all_fragments)} total records")
        logger.info(f"{'='*80}")
        
        return all_fragments

    def _process_single_block(self, block, index, total, doi, paper_dir):
        """处理单个 block 的辅助函数 (用于并行调用)"""
        logger.info(f"\n--- Block {index}/{total} ---")
        
        # 根据source_type准备输入 (返回结构化输入)
        if self.source_type == "text":
            input_content = self._format_text_input(block)
        elif self.source_type == "table":
            input_content = self._format_table_input(block, paper_dir)
        elif self.source_type == "figure":
            input_content = self._format_figure_input(block, paper_dir)
        else:
            return []
        
        if not input_content:
            return []
        
        # 三Agent流水线
        fragments = self._three_agent_pipeline(input_content, doi)
        
        if fragments:
            logger.info(f"  ✓ Block {index}: Extracted {len(fragments)} records")
            return fragments
        else:
            logger.info(f"  ℹ Block {index}: No records extracted")
            return []
    
    def _three_agent_pipeline(
        self,
        input_content: Dict[str, Any],
        doi: str
    ) -> List[Dict[str, Any]]:
        """
        执行三Agent流水线
        
        Args:
            input_content: 结构化输入内容
                - text: 文本内容
                - is_multimodal: 是否包含图像
                - image_path: 图像路径(可选)
                - caption: 图片/表格标题(可选)
                - markdown_table: Markdown表格(可选)
            doi: 论文DOI
            
        Returns:
            最终提取的fragments
        """
        # Stage 1: Extractor (Agent A)
        logger.info("  🔍 Stage 1: Extractor (high recall)")
        draft_json = self._agent_a_extract(input_content)
        
        if not draft_json:
            logger.warning("  ⚠ Extractor returned empty")
            return []
        
        logger.info(f"  ✓ Draft extracted: {len(draft_json)} records")
        
        # Stage 2: Reviewer (Agent B)
        logger.info("  📋 Stage 2: Reviewer (high precision)")
        review_report = self._agent_b_review(input_content, draft_json)
        
        logger.info(f"  ✓ Review completed")
        
        # Stage 3: Synthesizer (Agent C)
        logger.info("  🎯 Stage 3: Synthesizer (final version)")
        final_json = self._agent_c_synthesize(input_content, draft_json, review_report)
        
        if not final_json:
            logger.warning("  ⚠ Synthesizer returned empty, using draft")
            final_json = draft_json
        
        logger.info(f"  ✓ Final version: {len(final_json)} records")
        
        # 添加metadata
        for fragment in final_json:
            fragment["source_in_document"] = {
                "doi": doi,
                "source_type": self.source_type
            }
        
        return final_json
    
    def _agent_a_extract(self, input_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Agent A: Extractor - 高召回率提取
        
        角色：乐观的信息搜寻者
        目标：尽可能多地提取所有潜在信息
        
        Args:
            input_content: 结构化输入内容
            
        Returns:
            初稿JSON列表
        """
        # 构建提示词
        text_prompt = input_content.get("text", "")
        enhanced_prompt = f"""{self.extractor_prompt}

CRITICAL INSTRUCTION FOR THIS EXTRACTION:
You are Agent A - The Extractor. Your role is to be LIBERAL and INCLUSIVE in extraction.
- Extract ALL potential records, even if you're not 100% certain
- Prefer RECALL over PRECISION (better to extract too much than miss something)
- If in doubt, EXTRACT IT

Now process the following data:

{text_prompt}
"""
        
        # 构建消息
        messages = self._build_messages(enhanced_prompt, input_content)
        is_multimodal = input_content.get("is_multimodal", False)
        
        try:
            response = self.llm_client.chat(
                messages,
                is_multimodal=is_multimodal,
                json_mode=True,  # Enable JSON mode for structured output
                temperature=0.2,  # 稍高温度增加召回率
                max_tokens=8000,  # DeepSeek max limit is 8192
                thinking={"type": "enabled"}  # Extractor 使用 thinking mode 提高准确性
            )
            
            # 解析JSON
            fragments = self._parse_json_response(response)
            return fragments
            
        except Exception as e:
            logger.error(f"  ✗ Agent A failed: {e}")
            return []
    
    def _agent_b_review(
        self,
        input_content: Dict[str, Any],
        draft_json: List[Dict[str, Any]]
    ) -> str:
        """
        Agent B: Reviewer - 高精确率审核
        
        角色：严格的质量控制专家
        目标：找出所有错误和不一致
        
        Args:
            input_content: 结构化输入内容
            draft_json: Agent A的初稿
            
        Returns:
            审核报告（文本）
        """
        # 提取文本描述（用于审核报告）
        text_summary = input_content.get("text", "")
        
        review_prompt = f"""{self.reviewer_prompt}

=== ORIGINAL INPUT ===
{text_summary}

=== DRAFT JSON FROM EXTRACTOR ===
{json.dumps(draft_json, indent=2, ensure_ascii=False)}

=== YOUR TASK ===
Review the draft JSON against the original input. Your ONLY task is to find errors.

For EACH field in EACH record, check:
1. Is the value EXACTLY correct? (check numbers, units, spelling)
2. Is it consistent with the original input?
3. Are there any missing or extra values?

Output format:
- If a field is CORRECT: say "✓ field_name: correct"
- If a field has ERROR: say "✗ field_name: WRONG - [explain what's wrong and what's correct]"
- If a record should NOT exist: say "✗ RECORD #X: Should be removed - [explain why]"
- If a record is MISSING: say "⚠ MISSING RECORD: [describe what should be extracted]"

Be EXTREMELY STRICT. Even minor formatting issues should be flagged.
"""
        
        # 构建消息（reviewer不需要看图，只审核提取结果）
        messages = [{"role": "user", "content": review_prompt}]
        
        try:
            response = self.llm_client.chat(
                messages,
                is_multimodal=False,  # Reviewer 只看文本和JSON，不看图
                temperature=0.0,  # 低温度确保一致性
                max_tokens=8000,
                thinking=None  # 禁用 thinking mode 加速审核
            )
            
            return response
            
        except Exception as e:
            logger.error(f"  ✗ Agent B failed: {e}")
            return "Review failed - proceeding with draft"
    
    def _agent_c_synthesize(
        self,
        input_content: Dict[str, Any],
        draft_json: List[Dict[str, Any]],
        review_report: str
    ) -> List[Dict[str, Any]]:
        """
        Agent C: Synthesizer - 综合生成最终版本
        
        角色：经验丰富的主编
        目标：综合所有信息生成黄金标准
        
        Args:
            input_content: 结构化输入内容
            draft_json: Agent A的初稿
            review_report: Agent B的审核报告
            
        Returns:
            最终JSON列表
        """
        text_summary = input_content.get("text", "")
        
        synthesis_prompt = f"""{self.synthesizer_prompt}

=== ORIGINAL INPUT ===
{text_summary}

=== DRAFT JSON (from Extractor) ===
{json.dumps(draft_json, indent=2, ensure_ascii=False)}

=== REVIEW REPORT (from Reviewer) ===
{review_report}

=== YOUR TASK ===
You are the final authority. Generate the GOLD STANDARD JSON by:
1. Starting with the draft JSON
2. Carefully reading the review report
3. Fixing ALL errors mentioned in the review
4. Adding any missing records identified
5. Removing any records that shouldn't exist

Output ONLY the final, corrected JSON array. No explanations.
"""
        
        # 构建消息（synthesizer 可以重新看图来做最终判断）
        messages = self._build_messages(synthesis_prompt, input_content)
        is_multimodal = input_content.get("is_multimodal", False)
        
        try:
            response = self.llm_client.chat(
                messages,
                is_multimodal=is_multimodal,  # Synthesizer 重新看图来做最终判断
                json_mode=True,  # Enable JSON mode for final output
                temperature=0.1,  # 低温度确保准确性
                max_tokens=8000,  # DeepSeek max limit is 8192
                thinking=None  # Synthesizer 禁用 thinking mode，已有 review 指导
            )
            
            # 解析JSON
            fragments = self._parse_json_response(response)
            return fragments
            
        except Exception as e:
            logger.error(f"  ✗ Agent C failed: {e}")
            return draft_json  # 失败时返回draft
    
    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        解析LLM的JSON响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析后的JSON列表
        """
        # 移除markdown代码块
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        # 移除GLM-4V的特殊标记
        if "<|begin_of_box|>" in response:
            response = response.split("<|begin_of_box|>")[1].split("<|end_of_box|>")[0]
        
        response = response.strip()
        
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []
        except json.JSONDecodeError:
            logger.warning("  ⚠ Failed to parse JSON response")
            return []
    
    def _load_prompt(self, prompt_path: str) -> str:
        """加载提示词文件"""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt from {prompt_path}: {e}")
            return ""
    
    def _build_reviewer_prompt(self) -> str:
        """
        构建Reviewer Agent的提示词
        """
        return """You are Agent B - The Reviewer.

Your personality: EXTREMELY STRICT, DETAIL-ORIENTED, PERFECTIONIST

Your role:
- You are a quality control expert reviewing JSON extraction results
- Your ONLY job is to find errors - you do NOT generate new JSON
- You must check EVERY field in EVERY record against the original input
- Be HARSH - even tiny errors must be flagged

What counts as an error:
- Wrong numerical value (even by 0.01)
- Wrong unit (e.g., μM vs mM)
- Missing decimal point or extra zeros
- Typo in enzyme name or substrate
- Field value doesn't match original text
- Record extracted when it shouldn't be
- Record missing when it should be extracted

Your output should be a detailed review report, NOT JSON."""
    
    def _build_synthesizer_prompt(self) -> str:
        """
        构建Synthesizer Agent的提示词
        """
        return """You are Agent C - The Synthesizer (Chief Editor).

Your personality: EXPERIENCED, WISE, DECISIVE

Your role:
- You are the final authority on what gets published
- You receive: (1) original input, (2) draft JSON, (3) review report
- Your job: synthesize all information to produce the GOLD STANDARD
- You must fix ALL errors mentioned in the review
- You have final say on ambiguous cases

Guidelines:
- Start with the draft JSON as your base
- Apply ALL corrections from the review report
- Add missing records if reviewer identified them
- Remove incorrect records if reviewer flagged them
- For ambiguous cases, use your judgment based on the original input
- Output ONLY valid JSON - no explanations

Your output MUST be a valid JSON array, nothing else."""
    
    def _format_text_input(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化文本输入
        
        Args:
            block: 文本block
            
        Returns:
            结构化输入内容
        """
        text = block.get("text", "")
        
        if not text:
            return None
        
        return {
            "text": text,
            "is_multimodal": False
        }
    
    def _format_table_input(
        self,
        block: Dict[str, Any],
        paper_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        格式化表格输入（完整多模态支持）
        
        Args:
            block: 表格block
            paper_dir: 论文目录
            
        Returns:
            结构化输入内容，包含文本描述和可选图像
        """
        # 提取caption
        caption_raw = block.get("table_caption", [])
        if isinstance(caption_raw, list):
            caption = " ".join(caption_raw) if caption_raw else "No caption available"
        else:
            caption = caption_raw if caption_raw else "No caption available"
        
        # 提取Markdown表格
        markdown_table = block.get("markdown_table", "")
        
        # 构建文本描述
        text_content = f"Table Caption: {caption}\n\n"
        if markdown_table:
            text_content += f"Table Content (Markdown):\n{markdown_table}\n\n"
        
        # 检查是否有表格图像
        image_path = block.get("img_path") or block.get("image_path")

        result = {
            "text": text_content,
            "caption": caption,
            "markdown_table": markdown_table,
            "is_multimodal": False
        }

        # 如果有图像路径，添加多模态支持
        if image_path and isinstance(image_path, str) and image_path.strip():
            if not os.path.isabs(image_path):
                image_path = os.path.join(paper_dir, image_path)

            if image_path and os.path.exists(image_path):
                result["image_path"] = image_path
                result["is_multimodal"] = True
                logger.debug(f"  Table with image: {Path(image_path).name}")

        return result
    
    def _format_figure_input(
        self,
        block: Dict[str, Any],
        paper_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        格式化图片输入（完整多模态支持）
        
        Args:
            block: 图片block
            paper_dir: 论文目录
            
        Returns:
            结构化输入内容，必须包含图像
        """
        # 提取caption
        caption_raw = block.get("image_caption", [])
        if isinstance(caption_raw, list):
            caption = " ".join(caption_raw) if caption_raw else "No caption available"
        else:
            caption = caption_raw if caption_raw else "No caption available"
        
        # 获取图像路径
        image_path = block.get("img_path") or block.get("image_path")

        if not image_path or not isinstance(image_path, str) or not image_path.strip():
            logger.warning("  Figure block missing image path")
            return None

        # 转换为绝对路径
        if not os.path.isabs(image_path):
            image_path = os.path.join(paper_dir, image_path)

        if not os.path.exists(image_path):
            logger.warning(f"  Figure image not found: {image_path}")
            return None
        
        # 构建文本描述
        text_content = f"Figure Caption: {caption}"
        
        return {
            "text": text_content,
            "caption": caption,
            "image_path": image_path,
            "is_multimodal": True
        }
    
    def _build_messages(
        self,
        prompt: str,
        input_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        构建LLM消息格式
        
        Args:
            prompt: 提示词
            input_content: 输入内容
            
        Returns:
            消息列表
        """
        if not input_content.get("is_multimodal", False):
            # 纯文本消息
            return [{"role": "user", "content": prompt}]
        
        # 多模态消息
        image_path = input_content.get("image_path")
        
        if not image_path:
            logger.warning("  Multimodal flag set but no image_path provided")
            return [{"role": "user", "content": prompt}]
        
        # 使用与 FigureExtractor/TableExtractor 相同的格式
        return [
            {
                "role": "user",
                "text": prompt,
                "image_path": image_path
            }
        ]
