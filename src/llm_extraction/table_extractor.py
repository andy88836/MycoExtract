"""
Table Extractor with Multimodal Fusion and Verification

This extractor implements an innovative approach to table parsing using a multi-source
fusion strategy where the LLM acts as an intelligent arbiter:

1. **Data Preparation**: Parse HTML table to Markdown (structured attempt)
2. **Multi-Source Assembly**: Combine text context (caption + footnotes), 
   machine-parsed structure, and visual ground truth (image)
3. **LLM Arbitration**: The multimodal LLM fuses and verifies all sources,
   using the image as the primary source of truth for structure and content

This approach ensures maximum accuracy by leveraging:
- Visual verification against the original table image
- Structural guidance from automated parsing
- Semantic context from captions and footnotes
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import sys

import pandas as pd
from io import StringIO

# Add project root to path for imports
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.llm_clients.providers import BaseLLMClient

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extractor for table content using multimodal LLM verification and fusion.
    
    Core Innovation: Uses a three-source fusion approach where the LLM acts as
    an intelligent arbiter to combine information from:
    
    1. [Text Context]: Caption and footnotes for semantic understanding
    2. [Structured Attempt]: Machine-parsed Markdown for structural guidance
    3. [Visual Ground Truth]: Original table image as primary source of truth
    
    The LLM verifies, corrects, and fuses these sources to achieve maximum
    extraction accuracy.
    """
    
    def __init__(self, llm_client: BaseLLMClient, prompt_path: str = "prompts/prompts_extract_from_table.txt"):
        """
        Initialize the table extractor.
        
        Args:
            llm_client: Multimodal LLM client (must support images)
            prompt_path: Path to the extraction prompt template
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt(prompt_path)
        logger.info(f"Initialized TableExtractor with prompt from: {prompt_path}")
    
    def _load_prompt(self, prompt_path: str) -> str:
        """
        Load the extraction prompt template from file.
        
        This template contains the complete instructions for the LLM arbiter,
        including the fusion strategy and output schema.
        
        Returns:
            String content of the prompt template
        """
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
                logger.debug(f"Loaded prompt template ({len(prompt_content)} chars)")
                return prompt_content
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
    
    def extract(self, blocks: List[Dict[str, Any]], paper_image_dir: str, doi: str = "unknown") -> List[Dict[str, Any]]:
        """
        Extract enzyme kinetics data from tables using multimodal fusion.
        
        This method processes all table blocks from a parsed PDF, applying the
        fusion and verification strategy to each table.
        
        Args:
            blocks: List of content blocks from _content_list.json
            paper_image_dir: Directory containing extracted images for this paper
            doi: DOI of the paper for metadata injection
            
        Returns:
            List of extracted knowledge fragments (one per table row with data)
        """
        logger.info(f"Starting table extraction from {len(blocks)} blocks")
        
        all_fragments = []
        table_count = 0
        
        for i, block in enumerate(blocks, 1):
            if block.get("type") != "table":
                continue
            
            table_count += 1
            logger.info(f"Processing table {table_count} (block {i}/{len(blocks)})")
            
            try:
                fragments = self._extract_from_table(block, paper_image_dir, doi)
                all_fragments.extend(fragments)
                logger.info(f"  ✓ Extracted {len(fragments)} fragments from table {table_count}")
            except Exception as e:
                logger.error(f"  ✗ Error processing table {table_count}: {e}", exc_info=True)
                continue
        
        logger.info(f"Table extraction complete: {len(all_fragments)} fragments from {table_count} tables")
        return all_fragments
    
    def _extract_from_table(self, table_block: Dict[str, Any], image_dir: str, doi: str) -> List[Dict[str, Any]]:
        """
        Extract data from a single table using the Fusion and Verification strategy.
        
        **Core Innovation Implementation:**
        This method implements our three-source fusion approach:
        
        1. **Data Preparation**:
           - Parse HTML table to Markdown (structured attempt)
           - Load table image (visual ground truth)
           - Extract caption and footnotes (text context)
        
        2. **Dynamic Prompt Construction**:
           - Inject all three sources into the base prompt template
           - Create a cohesive instruction set for the LLM arbiter
        
        3. **Multimodal LLM Call**:
           - Send the complete prompt + image to the multimodal LLM
           - The LLM fuses and verifies information from all sources
           - Uses the image as ground truth for structure and content
        
        Args:
            table_block: Table block dictionary with metadata and HTML body
            image_dir: Directory containing table images
            doi: Paper DOI for metadata injection
            
        Returns:
            List of extracted knowledge fragments (JSON objects)
        """
        # ================================================================
        # STEP 1: DATA PREPARATION - Extract the three information sources
        # ================================================================
        
        # 1.1 Extract text context (caption and footnotes)
        caption = table_block.get("table_caption", [])
        if isinstance(caption, list):
            caption = " ".join(caption) if caption else ""
        
        footnotes = table_block.get("table_footnote", [])
        if isinstance(footnotes, list):
            footnotes = " ".join(footnotes) if footnotes else ""
        
        logger.debug(f"  Caption: {caption[:100]}...")
        logger.debug(f"  Footnotes: {footnotes[:100]}..." if footnotes else "  No footnotes")
        
        # 1.2 Parse HTML table to Markdown (structured attempt)
        html_body = table_block.get("table_body", "")
        markdown_table = self._parse_table_to_markdown(html_body)
        logger.debug(f"  Parsed table to Markdown ({len(markdown_table)} chars)")
        
        # 1.3 Get table image path (visual ground truth)
        image_path = self._get_table_image_path(table_block, image_dir)
        
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"  ⚠ Table image not found: {image_path}")
            logger.warning(f"  Falling back to text-only extraction (no visual verification)")
            return self._extract_without_image(caption, footnotes, markdown_table, doi)
        
        logger.debug(f"  Image path: {image_path}")
        
        # ================================================================
        # STEP 2: DYNAMIC PROMPT CONSTRUCTION - Fusion of all sources
        # ================================================================
        
        full_prompt = self._construct_fusion_prompt(caption, footnotes, markdown_table)
        logger.debug(f"  Constructed fusion prompt ({len(full_prompt)} chars)")
        
        # ================================================================
        # STEP 3: MULTIMODAL LLM CALL - Intelligent arbitration
        # ================================================================
        
        # Construct the message payload for multimodal input
        # The LLM will receive both the detailed prompt AND the table image
        messages = [
            {
                "role": "user",
                "text": full_prompt,
                "image_path": image_path
            }
        ]
        
        try:
            logger.debug(f"  Calling multimodal LLM for fusion and verification...")
            
            # Enable thinking mode with large token budget
            # Allocate enough tokens for both reasoning process AND JSON output
            response = self.llm_client.chat(
                messages, 
                is_multimodal=True, 
                temperature=0.1,
                max_tokens=16384  # Large budget for thinking + JSON output
            )
            
            logger.debug(f"  Received LLM response ({len(response) if response else 0} chars)")
            
            # Parse and validate the LLM's JSON response
            fragments = self._parse_llm_response(response, doi)
            
            return fragments
            
        except Exception as e:
            logger.error(f"  ✗ Multimodal extraction failed: {e}", exc_info=True)
            return []
    
    def _parse_table_to_markdown(self, html_body: str) -> str:
        """
        Parse HTML table to Pandas DataFrame and convert to Markdown.
        
        This creates the "Structured Attempt" - an automated parsing that may
        contain errors. The LLM will verify this against the visual ground truth.
        
        Args:
            html_body: HTML string containing table structure
            
        Returns:
            Markdown representation of the table (or error message)
        """
        if not html_body or not html_body.strip():
            logger.debug("  Empty table body")
            return "[Empty table - no HTML content provided]"
        
        try:
            # Wrap in proper HTML table tags if not present
            html_to_parse = html_body.strip()
            if not html_to_parse.startswith('<table'):
                html_to_parse = f"<table>{html_to_parse}</table>"
            
            # Parse with pandas
            dfs = pd.read_html(StringIO(html_to_parse))
            
            if not dfs:
                logger.warning("  pandas.read_html returned empty list")
                return "[Could not parse table - pandas returned no DataFrames]"
            
            # Take the first table
            df = dfs[0]
            
            # Convert to Markdown
            markdown = df.to_markdown(index=False)
            
            logger.debug(f"  ✓ Successfully parsed table to Markdown ({df.shape[0]} rows × {df.shape[1]} cols)")
            return markdown
            
        except Exception as e:
            logger.warning(f"  ⚠ Failed to parse HTML table with pandas: {e}")
            # Return raw HTML as fallback with length limit
            return f"[Raw HTML table - parsing failed]\n{html_body[:1000]}..."
    
    def _get_table_image_path(self, table_block: Dict[str, Any], image_dir: str) -> Optional[str]:
        """
        Determine the path to the table's image file (visual ground truth).
        
        This method tries multiple possible field names to locate the image path
        in the table block metadata.
        
        Args:
            table_block: Table block metadata from _content_list.json
            image_dir: Directory containing images for this paper
            
        Returns:
            Absolute path to table image, or None if not found
        """
        # Try multiple possible field names
        img_path = (
            table_block.get("img_path") or 
            table_block.get("image_path") or 
            table_block.get("table_img") or
            table_block.get("table_image")
        )
        
        if not img_path:
            logger.debug("  No image path found in table block metadata")
            return None
        
        # Construct full path if relative
        if not os.path.isabs(img_path):
            img_path = os.path.join(image_dir, img_path)
        
        # Normalize path for Windows/Unix compatibility
        img_path = os.path.normpath(img_path)
        
        return img_path
    
    def _construct_fusion_prompt(self, caption: str, footnotes: str, markdown_table: str) -> str:
        """
        Construct the complete prompt by dynamically injecting all three sources.
        
        **This is the key method for our Fusion and Verification strategy.**
        
        The base prompt template (loaded from file) contains the general instructions
        for the LLM arbiter. This method injects the specific table data:
        
        - [Text Context]: Caption and footnotes for semantic understanding
        - [Structured Attempt]: Machine-parsed Markdown for structural guidance
        - [Visual Ground Truth]: Mentioned as attached image
        
        The LLM will then fuse these sources, using the image as the primary
        source of truth.
        
        Args:
            caption: Table caption text
            footnotes: Table footnotes text
            markdown_table: Machine-parsed table in Markdown format
            
        Returns:
            Complete prompt string ready for multimodal LLM
        """
        # Start with the base template (contains full instructions and schema)
        full_prompt = self.prompt_template + "\n\n"
        
        full_prompt += "=" * 70 + "\n"
        full_prompt += "TABLE INFORMATION TO PROCESS\n"
        full_prompt += "=" * 70 + "\n\n"
        
        # Inject Text Context
        full_prompt += "**[Text Context]**\n\n"
        if caption:
            full_prompt += f"Caption: {caption}\n\n"
        else:
            full_prompt += "Caption: [No caption provided]\n\n"
        
        if footnotes:
            full_prompt += f"Footnotes: {footnotes}\n\n"
        else:
            full_prompt += "Footnotes: [No footnotes provided]\n\n"
        
        # Inject Structured Attempt
        full_prompt += "**[Structured Attempt]** (Machine-parsed Markdown)\n\n"
        full_prompt += "```markdown\n"
        full_prompt += markdown_table + "\n"
        full_prompt += "```\n\n"
        
        # Reference Visual Ground Truth
        full_prompt += "**[Visual Ground Truth]**\n\n"
        full_prompt += "(The original table image is attached to this message. "
        full_prompt += "Please use it as your PRIMARY source of truth for table structure and content.)\n\n"
        
        full_prompt += "=" * 70 + "\n"
        full_prompt += "Please analyze all three sources and extract the enzyme kinetics data.\n"
        full_prompt += "=" * 70 + "\n"
        
        return full_prompt
    
    def _extract_without_image(self, caption: str, footnotes: str, markdown_table: str, doi: str) -> List[Dict[str, Any]]:
        """
        Fallback extraction when table image is not available.
        
        This method is used when the visual ground truth cannot be loaded.
        It relies solely on text context and the machine-parsed Markdown structure.
        
        **Warning**: This approach is less reliable than the full fusion strategy,
        as it cannot verify the table structure visually.
        
        Args:
            caption: Table caption
            footnotes: Table footnotes
            markdown_table: Parsed table in Markdown format
            doi: Paper DOI
            
        Returns:
            List of extracted fragments (may be less accurate)
        """
        logger.info("  Performing TEXT-ONLY table extraction (no image verification)")
        
        # Construct a simplified prompt for text-only extraction
        prompt = "Extract enzyme kinetics data from the following table information.\n\n"
        
        prompt += "**[Text Context]**\n\n"
        if caption:
            prompt += f"Caption: {caption}\n\n"
        if footnotes:
            prompt += f"Footnotes: {footnotes}\n\n"
        
        prompt += "**[Table Structure]** (Machine-parsed Markdown)\n\n"
        prompt += "```markdown\n"
        prompt += markdown_table + "\n"
        prompt += "```\n\n"
        
        prompt += "Please extract the kinetic data and format your output as a JSON list "
        prompt += "following the schema defined in the main prompt template. "
        prompt += "Note: Since no image is available, rely on the parsed Markdown structure.\n"
        
        # Use text-only LLM call
        messages = [{"role": "user", "content": prompt}]
        
        try:
            logger.debug("  Calling text-only LLM...")
            response = self.llm_client.chat(messages, is_multimodal=False, temperature=0.1, max_tokens=16384)
            logger.debug(f"  Received response ({len(response)} chars)")
            
            fragments = self._parse_llm_response(response, doi)
            return fragments
            
        except Exception as e:
            logger.error(f"  ✗ Text-only table extraction failed: {e}", exc_info=True)
            return []
    
    def _parse_llm_response(self, response: str, doi: str) -> List[Dict[str, Any]]:
        """
        Parse and validate the LLM's JSON response.
        
        The LLM should return a JSON list of knowledge fragments. This method
        extracts the JSON from the response text, validates it, and injects
        metadata (DOI and source type).
        
        Handles multiple LLM output formats:
        - GLM-4V: <|begin_of_box|>[]<|end_of_box|>
        - Markdown: ```json [...] ```
        - Plain JSON: [...]
        
        Args:
            response: Raw LLM response string (may contain markdown, explanations, etc.)
            doi: Paper DOI to inject into each fragment's metadata
            
        Returns:
            List of validated knowledge fragments (JSON objects)
        """
        try:
            # Strategy 0: Handle GLM-4V special box format (including truncated)
            if '<|begin_of_box|>' in response:
                if '<|end_of_box|>' in response:
                    # Complete box format
                    box_match = re.search(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', response, re.DOTALL)
                    if box_match:
                        json_str = box_match.group(1).strip()
                        logger.debug("  ✓ Extracted JSON from GLM-4V box format")
                    else:
                        json_str = response
                else:
                    # Truncated box format (no closing tag)
                    logger.warning("  ⚠️ Detected truncated GLM-4V box format (missing <|end_of_box|>)")
                    parts = response.split('<|begin_of_box|>')
                    if len(parts) > 1:
                        json_str = parts[1].strip()
                        logger.debug(f"  ✓ Extracted {len(json_str)} chars from truncated box format")
                    else:
                        logger.error("  ✗ Failed to split box format, using full response")
                        json_str = response.strip()
            # Strategy 1: Try to extract JSON from markdown code block
            elif '```json' in response:
                json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.debug("  ✓ Extracted JSON from markdown code block")
                else:
                    # Try without 'json' keyword
                    json_match = re.search(r'```\s*(\[.*?\])\s*```', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        logger.debug("  ✓ Extracted JSON from markdown code block (no json keyword)")
                    else:
                        json_str = response
            # Strategy 2: Try to extract any JSON array
            elif '[' in response and ']' in response:
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug("  ✓ Extracted JSON array from response")
                else:
                    # Try to find simple array
                    json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        logger.debug("  ✓ Extracted simple JSON array")
                    else:
                        json_str = response.strip()
            else:
                # Strategy 3: Use entire response (might be pure JSON)
                json_str = response.strip()
                logger.debug("  Using entire response as JSON")
            
            # Clean up common issues
            json_str = json_str.strip()
            
            # Validate json_str is not empty
            if not json_str:
                logger.error("  ✗ Extracted JSON string is empty after processing")
                logger.error(f"  Original response length: {len(response)}")
                logger.error(f"  Response preview: {response[:200]}...")
                return []
            
            # Try to parse JSON
            fragments = self._tolerant_json_parse(json_str)
            
            # Handle empty array
            if isinstance(fragments, list) and len(fragments) == 0:
                logger.info("  ℹ LLM returned empty array (no kinetics data found in table)")
                return []
            
            # Ensure it's a list
            if not isinstance(fragments, list):
                if isinstance(fragments, dict):
                    fragments = [fragments]
                    logger.debug("  Converted single object to list")
                else:
                    logger.warning("  Response is not a list or dict, returning empty list")
                    return []
            
            # Inject metadata into each fragment and filter out non-dict items
            valid_fragments = []
            for i, fragment in enumerate(fragments):
                if not isinstance(fragment, dict):
                    logger.warning(f"  Fragment {i} is not a dict (type: {type(fragment)}), skipping: {str(fragment)[:100]}")
                    continue
                
                # Ensure source_in_document exists
                if "source_in_document" not in fragment:
                    fragment["source_in_document"] = {}
                
                # Inject DOI and source type
                fragment["source_in_document"]["doi"] = doi
                fragment["source_in_document"]["source_type"] = "table"
                valid_fragments.append(fragment)
            
            logger.debug(f"  ✓ Successfully parsed {len(valid_fragments)} valid fragments")
            return valid_fragments
            
        except json.JSONDecodeError as e:
            logger.error(f"  ✗ Failed to parse JSON from LLM response: {e}")
            logger.error(f"  Response snippet: {response[:500]}...")
            logger.debug(f"  Full response: {response}")
            return []
        except Exception as e:
            logger.error(f"  ✗ Unexpected error parsing LLM response: {e}", exc_info=True)
            logger.debug(f"  Full response: {response}")
            return []
    
    def _tolerant_json_parse(self, json_str: str) -> Any:
        """
        Parse JSON with tolerance for common GLM-4.5V formatting issues.
        
        Tries multiple strategies:
        1. Standard JSON parsing
        2. Remove trailing commas
        3. Fix common quote issues
        4. Handle truncated JSON (close unclosed brackets)
        
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
        
        # Strategy 4: Handle truncated JSON (close unclosed structures)
        try:
            # Count opening and closing brackets
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # If JSON is truncated, try to close it properly
            if open_braces > close_braces or open_brackets > close_brackets:
                logger.debug(f"  Detected truncated JSON: {{:{open_braces}/{close_braces}, [:{open_brackets}/{close_brackets}")
                fixed_str = json_str
                
                # Find the last complete JSON object
                # Strategy: truncate to the last valid '},' or '}'
                last_complete = max(
                    fixed_str.rfind('},'),
                    fixed_str.rfind('}')
                )
                
                if last_complete > 0:
                    # Truncate to last complete object
                    fixed_str = fixed_str[:last_complete + 1]
                    
                    # Remove any incomplete trailing comma
                    fixed_str = re.sub(r',\s*$', '', fixed_str)
                    
                    # Close the array if needed
                    if fixed_str.count('[') > fixed_str.count(']'):
                        fixed_str += ']'
                    
                    # Remove trailing commas before closing brackets
                    fixed_str = re.sub(r',(\s*\])', r'\1', fixed_str)
                    
                    result = json.loads(fixed_str)
                    logger.warning(f"  ⚠️ JSON was truncated, recovered {len(result) if isinstance(result, list) else 1} complete records")
                    return result
        except Exception as e:
            logger.debug(f"  Truncation fix failed: {e}")
        
        # Strategy 5: Last resort - raise original error
        return json.loads(json_str)  # This will raise the original error


if __name__ == "__main__":
    """
    Test the table extractor with the Fusion and Verification strategy.
    
    This test demonstrates:
    1. Loading a multimodal LLM client
    2. Creating a TableExtractor instance
    3. Processing a sample table with all three sources
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from src.llm_clients.providers import build_client
    
    print("=" * 70)
    print("TABLE EXTRACTOR - FUSION AND VERIFICATION STRATEGY")
    print("=" * 70)
    print()
    
    # Build a multimodal client (requires appropriate API key)
    print("Building multimodal LLM client...")
    try:
        # Try OpenAI first (most common)
        client = build_client("openai", "glm-4.5v")
        print(f"✓ Successfully built GLM-4.5v client")
    except Exception as e:
        print(f"✗ Failed to build OpenAI client: {e}")
        print("  Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Create extractor
    print("\nInitializing TableExtractor...")
    try:
        extractor = TableExtractor(client)
        print(f"✓ TableExtractor initialized successfully")
        print(f"  Prompt template: {len(extractor.prompt_template)} chars")
    except Exception as e:
        print(f"✗ Failed to initialize TableExtractor: {e}")
        exit(1)
    
    print("\n" + "=" * 70)
    print("READY FOR TABLE EXTRACTION")
    print("=" * 70)
    print()
    print("The extractor is ready to process tables with the following strategy:")
    print("  1. Parse HTML to Markdown (Structured Attempt)")
    print("  2. Load table image (Visual Ground Truth)")
    print("  3. Extract caption and footnotes (Text Context)")
    print("  4. Fuse all sources via multimodal LLM arbitration")
    print()
    print("To use in your pipeline:")
    print("  fragments = extractor.extract(blocks, image_dir, doi)")
    print()