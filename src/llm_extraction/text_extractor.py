"""
Text Extractor (Improved with Smart Chunking + Preprocessing)

Extracts enzyme kinetics data from text blocks and equations using LLMs.

NEW in v4:
- **Remove References section** (save 20-30% tokens, avoid extracting citations)
- LaTeX normalization ($0.33 μM$ → 0.33 μM)
- Smart chunking (avoid splitting kinetic parameter context)
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm_clients.providers import BaseLLMClient
from src.llm_extraction.text_preprocessor import preprocess_text

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Extractor for text-based content including paragraphs and equations.
    
    NEW: Implements smart text chunking to avoid LLM timeouts on large blocks.
    MAX_CHUNK_SIZE = 8000 chars (safe for most LLMs including GPT-5, gpt-5-nano)
    """
    
    def __init__(self, llm_client: BaseLLMClient, prompt_path: str = "prompts/prompts_extract_from_text.txt", max_workers: int = 2):
        """
        Initialize the text extractor.
        
        Args:
            llm_client: LLM client for text generation (does not need multimodal support)
            prompt_path: Path to the extraction prompt template
            max_workers: Maximum number of parallel workers for text extraction (default: 2, lowered to avoid API rate limits)
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompt_path)
        self.max_workers = max_workers
        
        # No fallback needed - using unified v4.2 prompt
        # (v4.2: 22 fields flat schema with is_wild_type)
        self.simplified_prompt = None
        
        logger.info(f"Initialized TextExtractor with prompt from: {prompt_path}")
        logger.info(f"  - Parallel workers: {max_workers}")
    
    def _load_prompt(self, prompt_path: str) -> str:
        """
        Load the extraction prompt template from file.
        
        Args:
            prompt_path: Path to prompt file
            
        Returns:
            Prompt template string
        """
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
    
    def extract(self, blocks: List[Dict[str, Any]], doi: str = "unknown") -> List[Dict[str, Any]]:
        """
        Extract enzyme kinetics data from text blocks.
        
        This method:
        1. Merges consecutive text blocks into coherent paragraphs (WITH SIZE LIMITS)
        2. Processes each paragraph and equation separately
        3. Calls the LLM for extraction
        4. Aggregates and returns all extracted fragments
        
        Args:
            blocks: List of content blocks from _content_list.json
            doi: DOI of the paper for metadata
            
        Returns:
            List of extracted knowledge fragments (JSON objects)
        """
        logger.info(f"Starting text extraction for {len(blocks)} blocks")
        
        # Merge consecutive text blocks (WITH SMART CHUNKING)
        merged_blocks = self._merge_text_blocks(blocks)
        logger.info(f"Merged into {len(merged_blocks)} coherent units")
        
        all_fragments = []
        futures = []
        
        # 使用 ThreadPoolExecutor 并行处理
        # worker 数量在构造函数中配置，默认5个
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, block in enumerate(merged_blocks, 1):
                block_type = block.get("type", "unknown")
                
                # Process text blocks
                if block_type == "text":
                    text_content = block.get("content", "").strip()
                    
                    if not text_content or len(text_content) < 50:
                        continue
                    
                    logger.info(f"Submitting text block {i}/{len(merged_blocks)} ({len(text_content)} chars)")
                    future = executor.submit(self._extract_from_text, text_content, doi)
                    futures.append(future)
                
                # Process equation blocks
                elif block_type == "equation":
                    latex_text = block.get("latex_text", "").strip()
                    
                    if not latex_text:
                        continue
                    
                    logger.info(f"Submitting equation block {i}/{len(merged_blocks)}")
                    context = f"Equation: {latex_text}"
                    future = executor.submit(self._extract_from_text, context, doi)
                    futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    fragments = future.result()
                    if fragments:
                        all_fragments.extend(fragments)
                except Exception as e:
                    logger.error(f"Error in parallel extraction: {e}")
        
        logger.info(f"Text extraction complete: {len(all_fragments)} fragments extracted")
        return all_fragments
    
    def _merge_text_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge consecutive text blocks with smart chunking to avoid timeouts.
        
        MAX_CHUNK_SIZE = 6000 chars per block (conservative limit for API stability)
        Large blocks are split at paragraph/sentence boundaries.
        
        Args:
            blocks: Original list of content blocks
            
        Returns:
            List of merged blocks (with large blocks split)
        """
        MAX_CHUNK_SIZE = 50000  # 增加到 50k，利用现代 LLM 的长窗口优势，减少调用次数
        
        merged = []
        current_text = []
        current_length = 0
        
        def flush_current_text():
            """将累积的文本拆分成合适大小的块"""
            if not current_text:
                return
            
            full_text = "\n\n".join(current_text)
            
            # 如果总长度小于阈值，直接添加
            if len(full_text) <= MAX_CHUNK_SIZE:
                merged.append({
                    "type": "text",
                    "content": full_text
                })
            else:
                # 需要拆分 - 按段落边界智能切分
                chunks = self._split_text_smartly(full_text, MAX_CHUNK_SIZE)
                for chunk in chunks:
                    merged.append({
                        "type": "text",
                        "content": chunk
                    })
                logger.info(f"  ✂️ Split large text block ({len(full_text)} chars) into {len(chunks)} chunks")
        
        # 合并相邻的 text 块
        for block in blocks:
            block_type = block.get("type", "")
            
            if block_type == "text":
                content = block.get("text", "").strip()
                if content:
                    # 检查是否需要提前 flush（避免单个 merged block 过大）
                    if current_length + len(content) > MAX_CHUNK_SIZE * 1.5:
                        flush_current_text()
                        current_text = []
                        current_length = 0
                    
                    current_text.append(content)
                    current_length += len(content)
            else:
                # 遇到非文本块 - flush 累积的文本
                flush_current_text()
                current_text = []
                current_length = 0
                
                # 添加非文本块
                if block_type in ["equation", "table", "image"]:
                    merged.append(block)
        
        # Flush 剩余文本
        flush_current_text()
        
        # 🔍 Log merged blocks summary
        logger.info("")
        logger.info("  📊 Merged blocks summary (with smart chunking):")
        for i, block in enumerate(merged, 1):
            if block.get("type") == "text":
                content = block.get("content", "")
                preview = content[:100].replace('\n', ' ')
                logger.info(f"    Block {i}: text ({len(content)} chars) - '{preview}...'")
            else:
                logger.info(f"    Block {i}: {block.get('type')}")
        logger.info("")
        
        return merged
    
    def _split_text_smartly(self, text: str, max_size: int) -> List[str]:
        """
        按段落边界智能拆分文本，避免截断句子。
        
        Args:
            text: 要拆分的长文本
            max_size: 每个块的最大字符数
            
        Returns:
            拆分后的文本块列表
        """
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        # 将文本按两个换行符（\n\n）分割成一个段落列表
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # 如果单个段落就超长，强制按句子切分
            if para_size > max_size:
                # 先 flush 当前chunk
                if current_chunk:
                    # 先将当前已累积的 chunk 保存起来
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # 按句子切分超长段落
                sentences = para.replace(". ", ".\n").split("\n")
                for sent in sentences:
                    if current_size + len(sent) > max_size:
                        if current_chunk:
                            chunks.append("\n\n".join(current_chunk))
                        current_chunk = [sent]
                        current_size = len(sent)
                    else:
                        current_chunk.append(sent)
                        current_size += len(sent)
            else:
                # 正常段落 - 检查是否会超长
                if current_size + para_size > max_size:
                    # 当前块已满，保存并开始新块
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_size = para_size
                else:
                    current_chunk.append(para)
                    current_size += para_size + 2  # +2 for "\n\n"
        
        # 添加最后一个块
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _extract_from_text(self, text_content: str, doi: str) -> List[Dict[str, Any]]:
        """
        Call the LLM to extract structured data from text.
        
        NEW v4 Strategy:
        1. Preprocess text (LaTeX normalization, OCR fixing)
        2. Try extraction with full prompt
        3. If empty response, fallback to simplified prompt
        4. If still empty, log warning and return []
        
        Args:
            text_content: Text to analyze (may contain LaTeX)
            doi: Paper DOI for metadata
            
        Returns:
            List of extracted knowledge fragments
        """
        # Step 1: Preprocess text (remove References, normalize LaTeX, clean whitespace)
        preprocessed_text = preprocess_text(text_content)
        
        # Log preprocessing changes if significant
        if len(preprocessed_text) != len(text_content):
            chars_removed = len(text_content) - len(preprocessed_text)
            logger.debug(f"  📝 Preprocessing removed {chars_removed} chars ({chars_removed / len(text_content) * 100:.1f}%)")
        
        # Step 2: Extract with LLM
        fragments = self._try_extract_with_prompt(
            preprocessed_text, 
            doi, 
            self.prompt_template,
            prompt_name="v4.2"
        )
        
        return fragments
    
    def _try_extract_with_prompt(
        self, 
        text_content: str, 
        doi: str, 
        prompt_template: str,
        prompt_name: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Try extraction with a specific prompt template.
        
        Args:
            text_content: Preprocessed text
            doi: Paper DOI
            prompt_template: Prompt template string
            prompt_name: Name for logging
            
        Returns:
            List of extracted fragments (empty if failed)
        """
        # Construct the full prompt
        prompt = prompt_template.replace("{TEXT_CONTENT}", text_content)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Call LLM (text-only, no images)
            # Use appropriate max_tokens for the model (DeepSeek max=8192, Kimi max=16384)
            response = self.llm_client.chat(
                messages, 
                is_multimodal=False, 
                temperature=0.1,
                max_tokens=8000  # 提升到8000，避免大块文本提取时输出被截断
            )
            
            # Check for empty response
            if not response or len(response.strip()) == 0:
                logger.warning(f"  ⚠️ {prompt_name} prompt: LLM returned empty response")
                return []
            
            # Parse JSON response
            fragments = self._parse_llm_response(response, doi)
            
            if fragments:
                logger.info(f"  ✅ {prompt_name} prompt: Extracted {len(fragments)} fragments")
            else:
                logger.warning(f"  ⚠️ {prompt_name} prompt: Response parse returned empty")
            
            return fragments
            
        except Exception as e:
            logger.error(f"  ❌ Error during {prompt_name} extraction: {e}")
            return []
    
    def _parse_llm_response(self, response: str, doi: str) -> List[Dict[str, Any]]:
        """
        Parse and validate the LLM's JSON response.
        
        Handles multiple LLM output formats:
        - GLM-4V: <|begin_of_box|>[]<|end_of_box|>
        - Markdown: ```json [...] ```
        - Plain JSON: [...]
        
        Args:
            response: Raw LLM response string
            doi: Paper DOI to inject into metadata
            
        Returns:
            List of validated knowledge fragments
        """
        try:
            # Strategy 0: Handle GLM-4V special box format
            if '<|begin_of_box|>' in response and '<|end_of_box|>' in response:
                box_match = re.search(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', response, re.DOTALL)
                if box_match:
                    json_str = box_match.group(1).strip()
                    logger.debug("  ✓ Extracted JSON from GLM-4V box format")
                else:
                    json_str = response
            # Strategy 1: Try to find JSON in markdown code blocks (MOST COMMON for GLM-4.6)
            elif '```json' in response or '```' in response:
                # Try with 'json' keyword first
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    logger.debug("  ✓ Found JSON in ```json code block")
                else:
                    # Try without 'json' keyword
                    json_match = re.search(r'```\s*([\s\S]*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        logger.debug("  ✓ Found JSON in ``` code block")
                    else:
                        json_str = response
            # Strategy 2: Try to find raw JSON array
            elif '[' in response and ']' in response:
                json_match = re.search(r'\[[\s\S]*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug("  ✓ Found raw JSON array")
                else:
                    json_str = response
            else:
                json_str = response
            
            # Clean up
            json_str = json_str.strip()
            
            # Debug: Show what we're trying to parse
            if not json_str:
                logger.error(f"  ✗ Empty JSON string after extraction!")
                logger.error(f"  Full response: {response[:500]}")
                return []
            
            # Parse JSON with tolerance for GLM-4.5V issues
            fragments = self._tolerant_json_parse(json_str)
            
            # Handle empty array
            if isinstance(fragments, list) and len(fragments) == 0:
                logger.info("  ℹ LLM returned empty array (no kinetics data found in text)")
                return []
            
            # Validate it's a list
            if not isinstance(fragments, list):
                if isinstance(fragments, dict):
                    logger.warning("  LLM response is not a list, wrapping in array")
                    fragments = [fragments]
                else:
                    logger.warning("  Response is not a list or dict, returning empty list")
                    return []
            
            # Inject DOI into each fragment's metadata and filter out non-dict items
            valid_fragments = []
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    logger.warning(f"  Fragment is not a dict (type: {type(fragment)}), skipping: {str(fragment)[:100]}")
                    continue
                if "source_in_document" not in fragment:
                    fragment["source_in_document"] = {}
                fragment["source_in_document"]["doi"] = doi
                fragment["source_in_document"]["source_type"] = "text"
                valid_fragments.append(fragment)
            
            logger.debug(f"  ✓ Successfully parsed {len(valid_fragments)} valid fragments from LLM response")
            return valid_fragments
            
        except json.JSONDecodeError as e:
            logger.error(f"  ✗ Failed to parse LLM response as JSON: {e}")
            logger.error(f"  Response snippet: {response[:500]}...")
            
            # Save failed response for debugging
            import hashlib
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            response_hash = hashlib.md5(response.encode()).hexdigest()[:8]
            debug_file = f"debug_parse_failed_{timestamp}_{response_hash}.txt"
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== JSON Parse Failed ===\n")
                    f.write(f"Error: {e}\n")
                    f.write(f"DOI: {doi}\n")
                    f.write(f"Timestamp: {timestamp}\n\n")
                    f.write(f"=== Full Response ({len(response)} chars) ===\n")
                    f.write(response)
                logger.warning(f"  💾 Saved failed response to: {debug_file}")
            except Exception as save_error:
                logger.debug(f"  Failed to save debug file: {save_error}")
            
            logger.debug(f"  Full response: {response}")
            return []
        except Exception as e:
            logger.error(f"  ✗ Unexpected error parsing response: {e}", exc_info=True)
            logger.debug(f"  Full response: {response}")
            return []
    
    def _tolerant_json_parse(self, json_str: str) -> Any:
        """
        Parse JSON with tolerance for common LLM formatting issues.
        
        Tries multiple strategies:
        1. Standard JSON parsing
        2. Remove trailing commas
        3. Fix missing commas between objects
        
        Args:
            json_str: JSON string to parse (already extracted from code blocks)
            
        Returns:
            Parsed JSON object/array
            
        Raises:
            json.JSONDecodeError: If all strategies fail
        """
        # NOTE: Code block extraction is already done in _parse_llm_response
        # This method receives pure JSON string
        cleaned_str = json_str.strip()
        
        # Strategy 1: Try standard parsing first
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            logger.debug("  Standard JSON parse failed, trying fixes...")
        
        # Strategy 2: Remove trailing commas before ] or }
        try:
            fixed_str = re.sub(r',(\s*[}\]])', r'\1', cleaned_str)
            result = json.loads(fixed_str)
            logger.debug("  ✓ Fixed trailing commas")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"  Trailing comma fix failed: {e}")
        
        # Strategy 3: Fix incomplete JSON (truncated response)
        try:
            # If JSON ends abruptly, try to close it properly
            trimmed = cleaned_str.rstrip()
            is_incomplete = (
                trimmed.endswith(',') or 
                not trimmed.endswith((']', '}')) or
                trimmed.count('"') % 2 != 0  # Odd number of quotes = unterminated string
            )
            
            if is_incomplete:
                logger.debug("  Detected incomplete JSON, attempting to close...")
                
                # Remove trailing incomplete content
                fixed_str = trimmed.rstrip(',')
                
                # If ends with incomplete string (odd quotes), remove it
                if fixed_str.count('"') % 2 != 0:
                    # Find last opening quote
                    last_quote = fixed_str.rfind('"')
                    if last_quote > 0:
                        # Check if before quote is a colon (field value) or comma (array item)
                        before_quote = fixed_str[:last_quote].rstrip()
                        if before_quote.endswith(':'):
                            # It's a field value, replace with null
                            fixed_str = before_quote + ' null'
                        else:
                            # It's likely part of previous content, just remove from last complete field
                            # Find last complete field by going back to previous comma or {
                            last_comma = fixed_str.rfind(',', 0, last_quote)
                            last_brace = fixed_str.rfind('{', 0, last_quote)
                            cutoff = max(last_comma, last_brace)
                            if cutoff > 0:
                                fixed_str = fixed_str[:cutoff+1].rstrip(',')
                
                # Count open/close brackets
                open_braces = fixed_str.count('{')
                close_braces = fixed_str.count('}')
                open_brackets = fixed_str.count('[')
                close_brackets = fixed_str.count(']')
                
                # Add missing closing brackets/braces
                fixed_str += '}' * (open_braces - close_braces)
                fixed_str += ']' * (open_brackets - close_brackets)
                
                result = json.loads(fixed_str)
                logger.warning(f"  ⚠️ Fixed incomplete JSON (removed truncated content, added {open_braces - close_braces} braces, {open_brackets - close_brackets} brackets)")
                return result
        except json.JSONDecodeError as e:
            logger.debug(f"  Incomplete JSON fix failed: {e}")
        
        # Strategy 4: Try to fix missing commas between objects (common GLM issue)
        try:
            fixed_str = re.sub(r'\}\s*\{', '},{', cleaned_str)
            fixed_str = re.sub(r',(\s*[}\]])', r'\1', fixed_str)  # Also remove trailing commas
            result = json.loads(fixed_str)
            logger.debug("  ✓ Fixed missing commas between objects")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"  Missing comma fix failed: {e}")
        
        # Strategy 5: Last resort - raise original error with full context
        logger.error(f"  ❌ All JSON fix strategies failed. JSON length: {len(cleaned_str)} chars")
        logger.error(f"  First 200 chars: {cleaned_str[:200]}")
        logger.error(f"  Last 200 chars: {cleaned_str[-200:]}")
        return json.loads(cleaned_str)  # This will raise the original error with line/col info
