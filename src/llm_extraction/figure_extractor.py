"""
Figure Extractor

Extracts enzyme kinetics data from figures and charts using multimodal LLMs.
Analyzes both the figure image and its caption for comprehensive extraction.

Optimized for GLM-4.5V and other vision-capable models.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from src.llm_clients.providers import BaseLLMClient

logger = logging.getLogger(__name__)


class FigureExtractor:
    """
    Extractor for figure content using multimodal vision models.
    """
    
    def __init__(self, llm_client: BaseLLMClient, prompt_path: str = "prompts/prompts_extract_from_figure.txt"):
        """
        Initialize the figure extractor.
        
        Args:
            llm_client: Multimodal LLM client (must support vision)
            prompt_path: Path to the extraction prompt template
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompt_path)
        logger.info(f"✓ Initialized FigureExtractor with model: {llm_client.model_name}")
        logger.info(f"✓ Loaded prompt template from: {prompt_path}")
    
    def _load_prompt(self, prompt_path: str) -> str:
        """Load the extraction prompt template from file."""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            logger.debug(f"  Loaded prompt template ({len(template)} chars)")
            return template
        except FileNotFoundError:
            logger.error(f"✗ Prompt file not found: {prompt_path}")
            raise
    
    def extract(self, blocks: List[Dict[str, Any]], paper_image_dir: str, doi: str = "unknown") -> List[Dict[str, Any]]:
        """
        Extract enzyme kinetics data from figures.
        
        Args:
            blocks: List of content blocks from _content_list.json
            paper_image_dir: Directory containing extracted images for this paper
            doi: DOI of the paper for metadata
            
        Returns:
            List of extracted knowledge fragments
        """
        logger.info(f"📊 Starting figure extraction from {len(blocks)} blocks for DOI: {doi}")
        
        all_fragments = []
        figure_contexts = self._identify_figure_contexts(blocks)
        
        logger.info(f"🔍 Identified {len(figure_contexts)} figure contexts to process")
        
        for i, fig_ctx in enumerate(figure_contexts, 1):
            logger.info(f"📷 Processing figure {i}/{len(figure_contexts)}: {Path(fig_ctx['image_path']).name}")
            
            try:
                fragments = self._extract_from_figure(fig_ctx, paper_image_dir, doi)
                all_fragments.extend(fragments)
                logger.info(f"  ✓ Extracted {len(fragments)} fragments from figure {i}")
            except Exception as e:
                logger.error(f"  ✗ Error processing figure {i}: {e}", exc_info=True)
                continue
        
        logger.info(f"✅ Figure extraction complete: {len(all_fragments)} total fragments from {len(figure_contexts)} figures")
        return all_fragments
    
    def _identify_figure_contexts(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify figure contexts (image + caption pairs) from blocks.
        
        Args:
            blocks: List of content blocks
            
        Returns:
            List of figure context dictionaries
        """
        figure_contexts = []
        
        for i, block in enumerate(blocks):
            if block.get("type") == "image":
                # Found an image block
                image_path = block.get("img_path") or block.get("image_path")
                
                if not image_path:
                    logger.debug(f"  Block {i}: Skipping image block without path")
                    continue
                
                # Look for associated caption (usually next block or in metadata)
                caption = block.get("image_caption", "")
                
                # Check next block for caption if not in metadata
                if not caption and i + 1 < len(blocks):
                    next_block = blocks[i + 1]
                    if next_block.get("type") == "text":
                        text_content = next_block.get("content", "")
                        # Heuristic: if text starts with "Figure" or "Fig", it's likely a caption
                        if re.match(r'^(Figure|Fig\.?)\s+\d+', text_content, re.IGNORECASE):
                            caption = text_content
                            logger.debug(f"  Block {i}: Found caption in next block")
                
                figure_contexts.append({
                    "image_path": image_path,
                    "caption": caption if caption else "No caption available",
                    "block_index": i
                })
                
                logger.debug(f"  Block {i}: Added figure context - {Path(image_path).name}")
        
        return figure_contexts
    
    def _extract_from_figure(self, fig_context: Dict[str, Any], image_dir: str, doi: str) -> List[Dict[str, Any]]:
        """
        Extract data from a single figure using multimodal LLM.
        
        Uses the same message format as table_extractor for consistency with GLM-4.5V.
        
        Args:
            fig_context: Figure context dictionary
            image_dir: Directory containing images
            doi: Paper DOI
            
        Returns:
            List of extracted fragments
        """
        # Get full image path
        image_path = fig_context["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_dir, image_path)
        
        if not os.path.exists(image_path):
            logger.warning(f"Figure image not found: {image_path}")
            return []
        
        # Get caption - handle both list and string formats
        caption_raw = fig_context.get("caption", "No caption available")
        if isinstance(caption_raw, list):
            caption = " ".join(caption_raw) if caption_raw else "No caption available"
        else:
            caption = caption_raw if caption_raw else "No caption available"
        
        # Construct prompt using template
        prompt = self.prompt_template.replace("{CAPTION}", caption)
        
        # Construct messages in the format optimized for multimodal models (especially GLM-4.5V)
        # Use "text" + "image_path" format (not "content")
        messages = [
            {
                "role": "user",
                "text": prompt,
                "image_path": image_path
            }
        ]
        
        try:
            logger.debug(f"  Calling multimodal LLM for figure analysis...")
            
            # Call with explicit multimodal parameters
            # - is_multimodal=True: Enable vision processing
            # - temperature=0.1: Low temperature for factual extraction
            # - max_tokens=16384: Large budget for thinking mode + JSON output
            response = self.llm_client.chat(
                messages, 
                is_multimodal=True, 
                temperature=0.1,
                max_tokens=16384  # Match table_extractor's token budget
            )
            
            logger.debug(f"  Received LLM response ({len(response)} chars)")
            
            # Parse and validate the LLM's JSON response
            fragments = self._parse_llm_response(response, doi)
            return fragments
            
        except Exception as e:
            logger.error(f"  ✗ Figure extraction failed: {e}", exc_info=True)
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
            # Strategy 1: Try to extract JSON from code block
            elif '```json' in response:
                json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.debug("  ✓ Found JSON in code block")
                else:
                    # Try without 'json' keyword
                    json_match = re.search(r'```\s*(\[.*?\])\s*```', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        logger.debug("  ✓ Found JSON in code block (no json keyword)")
                    else:
                        json_str = response
            # Strategy 2: Try to find any JSON array
            elif '[' in response and ']' in response:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug("  ✓ Found JSON array in response")
                else:
                    json_str = response
            else:
                # Strategy 3: Try entire response
                json_str = response.strip()
                logger.debug("  Using entire response as JSON")
            
            # Clean up
            json_str = json_str.strip()
            
            # Parse JSON with tolerance for GLM-4.5V issues
            fragments = self._tolerant_json_parse(json_str)
            
            # Handle empty array
            if isinstance(fragments, list) and len(fragments) == 0:
                logger.info("  ℹ LLM returned empty array (no kinetics data found in figure)")
                return []
            
            # Ensure it's a list
            if not isinstance(fragments, list):
                if isinstance(fragments, dict):
                    fragments = [fragments]
                    logger.debug("  Converted single fragment to list")
                else:
                    logger.warning("  Response is not a list or dict, returning empty list")
                    return []
            
            # Inject metadata (DOI and source type) and filter out non-dict items
            valid_fragments = []
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    logger.warning(f"  Fragment is not a dict (type: {type(fragment)}), skipping: {str(fragment)[:100]}")
                    continue
                if "source_in_document" not in fragment:
                    fragment["source_in_document"] = {}
                fragment["source_in_document"]["doi"] = doi
                fragment["source_in_document"]["source_type"] = "figure"
                valid_fragments.append(fragment)
            
            logger.debug(f"  ✓ Parsed {len(valid_fragments)} valid fragments from LLM response")
            return valid_fragments
            
        except json.JSONDecodeError as e:
            logger.error(f"  ✗ Failed to parse figure extraction response as JSON: {e}")
            logger.error(f"  Response snippet: {response[:500]}...")
            logger.debug(f"  Full response: {response}")
            return []
        except Exception as e:
            logger.error(f"  ✗ Unexpected error parsing response: {e}", exc_info=True)
            logger.debug(f"  Full response: {response}")
            return []
    
    def _tolerant_json_parse(self, json_str: str) -> Any:
        """
        Parse JSON with tolerance for common GLM-4.5V formatting issues.
        
        Tries multiple strategies:
        1. Standard JSON parsing
        2. Remove trailing commas
        3. Fix missing commas between objects
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            Parsed JSON object/array
            
        Raises:
            json.JSONDecodeError: If all strategies fail
        """
        # Strategy 1: Try standard parsing first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.debug("  Standard JSON parse failed, trying fixes...")
        
        # Strategy 2: Remove trailing commas before ] or }
        try:
            fixed_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            result = json.loads(fixed_str)
            logger.debug("  ✓ Fixed trailing commas")
            return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Try to fix missing commas between objects (common GLM issue)
        try:
            fixed_str = re.sub(r'\}\s*\{', '},{', json_str)
            fixed_str = re.sub(r',(\s*[}\]])', r'\1', fixed_str)  # Also remove trailing commas
            result = json.loads(fixed_str)
            logger.debug("  ✓ Fixed missing commas between objects")
            return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Last resort - raise original error
        return json.loads(json_str)  # This will raise the original error


if __name__ == "__main__":
    """
    Test the figure extractor with GLM-4.5V.
    
    Usage:
        $env:ZHIPUAI_API_KEY = "your-key"
        python -m src.llm_extraction.figure_extractor
    """
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from src.llm_clients.providers import build_client
    
    print("=" * 70)
    print("  🧪 FigureExtractor 测试")
    print("=" * 70)
    print()
    
    # Check API key
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 ZHIPUAI_API_KEY 环境变量")
        print("   请先设置: $env:ZHIPUAI_API_KEY = 'your-key'")
        sys.exit(1)
    
    print(f"✅ API Key 已设置: {api_key[:20]}...")
    print()
    
    # Build multimodal client (GLM-4.5V is recommended for vision tasks)
    print("📦 构建 GLM-4.5V 客户端...")
    client = build_client("zhipuai", "glm-4.5v")
    print(f"✅ 客户端构建成功: {client.model_name}")
    print()
    
    # Create extractor
    print("🔧 初始化 FigureExtractor...")
    extractor = FigureExtractor(client)
    print("✅ FigureExtractor 初始化成功")
    print()
    
    print("=" * 70)
    print("✓ 所有测试通过！")
    print("=" * 70)