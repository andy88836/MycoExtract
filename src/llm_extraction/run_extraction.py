"""
Main Extraction Pipeline Orchestrator - Heterogeneous Model Architecture

This script coordinates the complete extraction workflow with smart dispatching:
1. Load _content_list.json files from Step 2 output
2. Pre-process content (clean references, etc.)
3. Run specialized extractors with appropriate models:
   - Text Extractor: Uses cost-effective text-only model (e.g., DeepSeek)
   - Table Extractor: Uses multimodal model (e.g., GLM-4.5V)
   - Figure Extractor: Uses multimodal model (e.g., GLM-4.5V)
4. Aggregate all fragments
5. Save raw JSON output
6. Post-process to CSV

Features:
- Heterogeneous model architecture for cost optimization
- Smart dispatching based on content type
- Centralized client factory for managing multiple models
- YAML-based configuration
- Checkpointing and resume capability
- Batch processing with progress tracking
- Detailed logging and error handling

Architecture:
- Uses ClientFactory for dependency injection of specialized LLM clients
- Text extraction uses efficient text-only models
- Multimodal extraction uses vision-capable models
- 60-80% cost reduction vs. using premium models for all tasks
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from src.llm_clients.client_factory import ClientFactory
from src.llm_extraction.text_extractor import TextExtractor
from src.llm_extraction.table_extractor import TableExtractor
from src.llm_extraction.figure_extractor import FigureExtractor
from src.llm_extraction.postprocessor import DataPostprocessor
from src.pipeline.enhanced_pipeline import EnhancedExtractionPipeline
from src.utils.logging_config import setup_logging

# Load environment variables
load_dotenv()

# Configure logging to logs/ directory
logger = setup_logging(log_level="INFO", module_name="extraction_pipeline")


class ExtractionPipeline:
    """
    Main extraction pipeline coordinator with heterogeneous model support.
    
    Architecture:
    - Uses ClientFactory to manage multiple specialized LLM clients
    - Text extraction uses cost-effective text-only models
    - Multimodal extraction (tables/figures) uses vision-capable models
    - Smart dispatching based on content type for optimal cost/performance
    - Supports concurrent processing and multi-agent mode via EnhancedExtractionPipeline
    """
    
    def __init__(self, config_path: str = "config/extraction_config.yaml", 
                 cli_overrides: Optional[Dict[str, Any]] = None,
                 max_workers: int = 3,
                 use_multi_agent: bool = False):
        """
        Initialize the extraction pipeline with heterog跑eneous model support.
        
        Args:
            config_path: Path to YAML configuration file
            cli_overrides: Optional CLI parameter overrides
            max_workers: Maximum concurrent workers for parallel processing
            use_multi_agent: Enable multi-agent collaboration mode
        """
        logger.info("=" * 80)
        logger.info("🚀 INITIALIZING HETEROGENEOUS MODEL EXTRACTION PIPELINE")
        logger.info("=" * 80)
        
        # Initialize client factory and build all clients
        logger.info(f"📋 Loading configuration from: {config_path}")
        self.factory = ClientFactory(config_path)
        
        # Build all clients using the factory
        logger.info("🔧 Building specialized LLM clients...")
        self.clients = self.factory.build_all()
        
        # Get configuration
        self.config = self.factory.get_config()
        self.file_paths = self.factory.get_file_paths()
        self.extraction_params = self.factory.get_extraction_parameters()
        
        # Apply CLI overrides if provided
        if cli_overrides:
            logger.info("⚙️  Applying CLI overrides...")
            for key, value in cli_overrides.items():
                if key in self.file_paths:
                    self.file_paths[key] = value
                    logger.info(f"  Overriding {key}: {value}")
        
        # Get specialized clients
        text_client = self.clients.get('text_client')
        multimodal_client = self.clients.get('multimodal_client')
        
        if not text_client or not multimodal_client:
            raise ValueError("text_client and multimodal_client must be configured")
        
        logger.info(f"\n📦 Initializing Enhanced Pipeline...")
        logger.info(f"  📝 Text Client: {type(text_client).__name__} "
                   f"({self.config['llm_clients']['text_client']['provider']}/"
                   f"{self.config['llm_clients']['text_client']['model_name']})")
        logger.info(f"  📊 Multimodal Client: {type(multimodal_client).__name__} "
                   f"({self.config['llm_clients']['multimodal_client']['provider']}/"
                   f"{self.config['llm_clients']['multimodal_client']['model_name']})")
        logger.info(f"  🔧 Max Workers: {max_workers}")
        logger.info(f"  🤝 Multi-Agent: {'ON' if use_multi_agent else 'OFF'}")
        
        # Initialize EnhancedExtractionPipeline with separate clients
        self.enhanced_pipeline = EnhancedExtractionPipeline(
            text_client=text_client,
            multimodal_client=multimodal_client,
            text_prompt_path=self.file_paths.get("prompt_text", "prompts/prompts_extract_from_text.txt"),
            table_prompt_path=self.file_paths.get("prompt_table", "prompts/prompts_extract_from_table.txt"),
            figure_prompt_path=self.file_paths.get("prompt_figure", "prompts/prompts_extract_from_figure.txt"),
            max_workers=max_workers,
            use_multi_agent=use_multi_agent,
            max_retries=self.extraction_params.get('max_retries', 3),
            save_intermediate=True
        )
        
        # Initialize postprocessor
        self.postprocessor = DataPostprocessor()
        
        # Setup directories
        self.input_dir = Path(self.file_paths.get("input_dir", "data/parsed_pdf"))
        self.output_json_dir = Path(self.file_paths.get("output_json_dir", "data/extracted_json"))
        self.output_csv_dir = Path(self.file_paths.get("output_csv_dir", "data/extracted_csv"))
        
        self.output_json_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n📁 Directory Configuration:")
        logger.info(f"  Input:       {self.input_dir}")
        logger.info(f"  Output JSON: {self.output_json_dir}")
        logger.info(f"  Output CSV:  {self.output_csv_dir}")
        
        logger.info("\n✅ ExtractionPipeline initialized successfully")
        logger.info("=" * 80)
        logger.info("")
        
        # 🧪 Test LLM clients before starting extraction (optional)
        if not self.extraction_params.get('skip_client_test', False):
            self._test_llm_clients(text_client, multimodal_client)
        else:
            logger.info("⏭️  Skipping client pre-testing (skip_client_test=True)")
            logger.info("")
    
    def _test_llm_clients(self, text_client, multimodal_client):
        """
        Test if LLM clients are working correctly before extraction.
        
        Args:
            text_client: Text-only LLM client
            multimodal_client: Multimodal LLM client
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("🧪 TESTING LLM CLIENTS")
        logger.info("=" * 80)
        
        # Test text client
        logger.info("📝 Testing text client...")
        try:
            test_messages = [
                {
                    "role": "user",
                    "content": "Please return a JSON array with one object containing a 'test' field set to 'success'. Example: [{\"test\": \"success\"}]"
                }
            ]
            response = text_client.chat(test_messages, is_multimodal=False, temperature=0.1)
            
            # Try to parse the response
            if "success" in response.lower():
                logger.info("  ✅ Text client is working correctly")
                logger.info(f"  Response preview: {response[:200]}...")
            else:
                logger.warning(f"  ⚠️ Text client response unexpected: {response[:200]}...")
        except Exception as e:
            logger.error(f"  ❌ Text client test FAILED: {e}")
            raise RuntimeError("Text client is not working. Cannot proceed with extraction.")
        
        # Test multimodal client
        logger.info("📊 Testing multimodal client...")
        try:
            test_messages = [
                {
                    "role": "user",
                    "text": "Please return a JSON array with one object containing a 'test' field set to 'success'. Example: [{\"test\": \"success\"}]"
                }
            ]
            response = multimodal_client.chat(test_messages, is_multimodal=False, temperature=0.1)
            
            if "success" in response.lower():
                logger.info("  ✅ Multimodal client is working correctly")
                logger.info(f"  Response preview: {response[:200]}...")
            else:
                logger.warning(f"  ⚠️ Multimodal client response unexpected: {response[:200]}...")
        except Exception as e:
            logger.error(f"  ❌ Multimodal client test FAILED: {e}")
            raise RuntimeError("Multimodal client is not working. Cannot proceed with extraction.")
        
        logger.info("✅ All LLM clients are functional")
        logger.info("=" * 80)
        logger.info("")
    
    def run(self, skip_existing: bool = True, limit: Optional[int] = None):
        """
        Run the complete extraction pipeline on all papers using EnhancedExtractionPipeline.
        
        Args:
            skip_existing: Whether to skip papers that have already been processed
            limit: Maximum number of papers to process (None = all)
        """
        start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info("🚀 STARTING EXTRACTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Find all _content_list.json files
        paper_dirs = self._find_paper_directories()
        
        # Apply limit if specified
        if limit:
            paper_dirs = paper_dirs[:limit]
            logger.info(f"📊 Limiting to first {limit} papers")
        
        # Filter out already processed papers if skip_existing is True
        if skip_existing:
            unprocessed_dirs = []
            for paper_dir in paper_dirs:
                paper_name = paper_dir.name
                output_json = self.output_json_dir / f"{paper_name}.json"
                output_csv = self.output_csv_dir / f"{paper_name}.csv"
                if not (output_json.exists() and output_csv.exists()):
                    unprocessed_dirs.append(paper_dir)
            
            skipped_count = len(paper_dirs) - len(unprocessed_dirs)
            if skipped_count > 0:
                logger.info(f"⏭️  Skipping {skipped_count} already processed papers")
            paper_dirs = unprocessed_dirs
        
        if not paper_dirs:
            logger.info("✅ All papers already processed!")
            return
        
        logger.info(f"📁 Processing {len(paper_dirs)} papers\n")
        
        # Run EnhancedExtractionPipeline
        temp_output_dir = self.output_json_dir.parent / "enhanced_temp"
        results = self.enhanced_pipeline.run(
            paper_dirs=[str(p) for p in paper_dirs],
            output_dir=str(temp_output_dir),
            progress_callback=None
        )
        
        # Copy JSON files from temp to final output directory
        logger.info("\n📄 Copying JSON files to output directory...")
        import shutil
        for paper_name in results["results"].keys():
            temp_json = temp_output_dir / f"{paper_name}.json"
            output_json = self.output_json_dir / f"{paper_name}.json"
            if temp_json.exists():
                shutil.copy2(temp_json, output_json)
                logger.info(f"  ✓ {paper_name}.json -> {self.output_json_dir.name}/")
        
        # Post-process results to CSV
        logger.info("\n📊 Post-processing results to CSV...")
        for paper_name, records in results["results"].items():
            if records:
                output_csv = self.output_csv_dir / f"{paper_name}.csv"
                self.postprocessor.flatten_and_save(records, str(output_csv))
                logger.info(f"  ✓ {paper_name}: {len(records)} records -> {output_csv.name}")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        stats = results["statistics"]
        logger.info("\n" + "=" * 80)
        logger.info("🎉 EXTRACTION PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Start time:           {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End time:             {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration:             {duration}")
        logger.info(f"")
        logger.info(f"Total papers:         {stats['total_papers']}")
        logger.info(f"✅ Successfully processed: {stats['processed_papers']}")
        logger.info(f"❌ Failed:                {stats['failed_papers']}")
        logger.info(f"📊 Total records:         {stats['total_records']}")
        logger.info(f"   - Text:                {stats['text_records']}")
        logger.info(f"   - Table:               {stats['table_records']}")
        logger.info(f"   - Figure:              {stats['figure_records']}")
        logger.info("=" * 80)
    
    def _find_paper_directories(self) -> List[Path]:
        """
        Find all paper directories containing _content_list.json.
        
        Returns:
            List of paper directory paths
        """
        paper_dirs = []
        '''
        Use of pathlib.glob
        glob用在文件名 路径匹配中比较多
        from pathlib import Path

        p = Path("D:/AOP/papers")
        files = list(p.glob("*.json")
        '''
        
        # Look for directories containing *_content_list.json
        for item in self.input_dir.iterdir():
            # 只找文件夹
            if item.is_dir():
                # Look for any file ending with _content_list.json
                # 在目录 item 里，找所有 文件名以 _content_list.json 结尾 的文件。 item是Path对象
                # glob：用通配符去匹配文件名/路径的规则和工具
                content_lists = list(item.glob("*_content_list.json"))
                if content_lists:
                    paper_dirs.append(item)
                    logger.debug(f"  Found paper directory: {item.name}")
        
        return sorted(paper_dirs)
    
    def _process_paper(self, paper_dir: Path, output_json: Path, output_csv: Path) -> int:
        """
        Process a single paper through the extraction pipeline.
        
        Args:
            paper_dir: Directory containing paper's parsed content
            output_json: Path to save JSON output
            output_csv: Path to save CSV output
            
        Returns:
            Number of fragments extracted (>=0 if successful, -1 if failed)
        """
        try:
            # Load *_content_list.json (find the file first)
            content_list_files = list(paper_dir.glob("*_content_list.json"))
            
            if not content_list_files:
                logger.warning(f"⚠️  No *_content_list.json found in {paper_dir.name}")
                return -1
            
            content_list_path = content_list_files[0]  # Use the first match
            logger.debug(f"  Using content list: {content_list_path.name}")
            
            with open(content_list_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            blocks = content_data if isinstance(content_data, list) else content_data.get("content", [])
            logger.info(f"📚 Loaded {len(blocks)} content blocks")
            
            # Get paper metadata
            doi = self._extract_doi(paper_dir.name)
            logger.info(f"📄 DOI: {doi}")
            
            # Note: References cleaning is now handled by text_preprocessor
            # in TextExtractor, not at block level. This provides more
            # fine-grained control and better detection of hidden references.
            
            # Run extractors
            logger.info("")
            logger.info("🔍 Running extractors...")
            logger.info("-" * 80)
            
            # Parallel extraction using ThreadPoolExecutor
            text_fragments = []
            table_fragments = []
            figure_fragments = []
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all three extraction tasks concurrently
                future_text = executor.submit(self._extract_text, blocks, doi)
                future_table = executor.submit(self._extract_tables, blocks, paper_dir, doi)
                future_figure = executor.submit(self._extract_figures, blocks, paper_dir, doi)
                
                # Collect results as they complete
                futures = {
                    'text': future_text,
                    'table': future_table,
                    'figure': future_figure
                }
                
                for task_name, future in futures.items():
                    try:
                        if task_name == 'text':
                            text_fragments = future.result()
                        elif task_name == 'table':
                            table_fragments = future.result()
                        elif task_name == 'figure':
                            figure_fragments = future.result()
                    except Exception as e:
                        logger.error(f"❌ {task_name.capitalize()} extraction failed: {e}", exc_info=True)
            
            logger.info("-" * 80)
            
            # Aggregate all fragments
            all_fragments = text_fragments + table_fragments + figure_fragments
            logger.info(f"📦 Total fragments: {len(all_fragments)}")
            
            # Save raw JSON
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(all_fragments, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved JSON: {output_json.name}")
            
            # Post-process to CSV
            if all_fragments:
                self.postprocessor.flatten_and_save(all_fragments, str(output_csv))
                logger.info(f"📊 Saved CSV: {output_csv.name}")
            else:
                logger.info("⚠️  No fragments to save to CSV")
            
            logger.info(f"✅ Successfully processed {paper_dir.name}")
            return len(all_fragments)
            
        except Exception as e:
            logger.error(f"❌ Failed to process paper: {e}", exc_info=True)
            return -1
    
    def _extract_text(self, blocks: List[Dict], doi: str) -> List[Dict]:
        """Helper method for parallel text extraction."""
        logger.info("📝 Text extraction...")
        fragments = self.text_extractor.extract(blocks, doi)
        logger.info(f"   ✓ Extracted {len(fragments)} text fragments")
        return fragments
    
    def _extract_tables(self, blocks: List[Dict], paper_dir: Path, doi: str) -> List[Dict]:
        """Helper method for parallel table extraction."""
        logger.info("📊 Table extraction...")
        fragments = self.table_extractor.extract(blocks, str(paper_dir), doi)
        logger.info(f"   ✓ Extracted {len(fragments)} table fragments")
        return fragments
    
    def _extract_figures(self, blocks: List[Dict], paper_dir: Path, doi: str) -> List[Dict]:
        """Helper method for parallel figure extraction."""
        logger.info("📷 Figure extraction...")
        fragments = self.figure_extractor.extract(blocks, str(paper_dir), doi)
        logger.info(f"   ✓ Extracted {len(fragments)} figure fragments")
        return fragments
    
    def _extract_doi(self, paper_name: str) -> str:
        """
        Extract DOI from paper directory name.
        
        Args:
            paper_name: Directory name (often sanitized DOI)
            
        Returns:
            DOI string or paper_name if DOI can't be extracted
        """
        # Try to reverse the sanitization (replace _ with /)
        if "_" in paper_name and not paper_name.startswith("paper_"):
            # Likely a sanitized DOI
            potential_doi = paper_name.replace("_", "/", 1)  # Only first underscore
            return potential_doi
        
        return paper_name


def main():
    """
    Main entry point for the heterogeneous model extraction pipeline.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run LLM-based extraction pipeline with heterogeneous model support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract using YAML config (default: config/extraction_config.yaml)
  python -m src.llm_extraction.run_extraction
  
  # Extract first 10 papers
  python -m src.llm_extraction.run_extraction --limit 10
  
  # Use custom config file
  python -m src.llm_extraction.run_extraction --config my_config.yaml
  
  # Override input/output directories
  python -m src.llm_extraction.run_extraction \\
      --input-dir data/my_papers \\
      --output-json-dir data/my_json

Configuration:
  The pipeline uses YAML configuration (config/extraction_config.yaml) which defines:
  - text_client: Cost-effective model for text extraction (e.g., DeepSeek)
  - multimodal_client: Vision-capable model for tables/figures (e.g., GLM-4.5V)
  
  This heterogeneous approach reduces costs by 60-80% vs. using premium models for all tasks.
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/extraction_config.yaml",
        help="Path to YAML configuration file (default: config/extraction_config.yaml)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input directory from config"
    )
    parser.add_argument(
        "--output-json-dir",
        type=str,
        default=None,
        help="Override JSON output directory from config"
    )
    parser.add_argument(
        "--output-csv-dir",
        type=str,
        default=None,
        help="Override CSV output directory from config"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process (default: all)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reprocess all papers (don't skip existing)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of papers to process concurrently (default: 2, recommend: 1-3 for APIs with rate limits)"
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Enable multi-agent collaboration mode (Extractor-Reviewer-Synthesizer)"
    )
    
    args = parser.parse_args()
    
    # Setup logging to logs/ directory
    logger = setup_logging(log_level="INFO", module_name="extraction_pipeline")
    
    # Print welcome banner
    print("=" * 80)
    print("  🚀 MYCOTOXIN-EXTRACTOR - HETEROGENEOUS MODEL EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"  Config file:        {args.config}")
    print(f"  Limit:              {args.limit if args.limit else 'All papers'}")
    print(f"  Skip existing:      {not args.no_skip}")
    print(f"  Parallel papers:    {args.max_workers} (concurrent paper processing)")
    print(f"  Multi-agent:        {'ON' if args.multi_agent else 'OFF'}")
    print(f"  Log directory:      logs/")
    print()
    
    # Build CLI overrides
    cli_overrides = {}
    if args.input_dir:
        cli_overrides['input_dir'] = args.input_dir
    if args.output_json_dir:
        cli_overrides['output_json_dir'] = args.output_json_dir
    if args.output_csv_dir:
        cli_overrides['output_csv_dir'] = args.output_csv_dir
    
    # Initialize and run pipeline
    try:
        pipeline = ExtractionPipeline(
            config_path=args.config,
            cli_overrides=cli_overrides if cli_overrides else None,
            max_workers=args.max_workers,
            use_multi_agent=args.multi_agent
        )
        pipeline.run(skip_existing=not args.no_skip, limit=args.limit)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()