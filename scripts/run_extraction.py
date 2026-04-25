#!/usr/bin/env python3
"""
MycoExtract: 霉菌毒素降解酶数据提取流水线

Multi-Agent LLM Pipeline for Automated Extraction of
Mycotoxin-Degrading Enzyme Kinetics Data

Usage:
    python scripts/run_extraction.py --input-dir data/papers --output-dir results
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.enhanced_pipeline import EnhancedExtractionPipeline
from src.llm_clients import build_client
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_extraction(input_dir: str, output_dir: str = None, limit: int = None,
                   config_file: str = None, max_workers: int = 2):
    """
    运行提取流程

    Args:
        input_dir: 输入目录（包含解析后的论文文件夹）
        output_dir: 输出目录
        limit: 限制处理的论文数量（用于测试）
        config_file: 配置文件路径
        max_workers: 最大并发数（GLM-4.7建议2）
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"❌ 输入目录不存在: {input_path}")
        return

    # 获取所有论文文件夹
    paper_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    logger.info(f"📂 找到 {len(paper_dirs)} 篇论文")

    # 限制数量
    if limit and limit < len(paper_dirs):
        logger.info(f"🔢 限制处理数量: {limit} / {len(paper_dirs)}")
        paper_dirs = paper_dirs[:limit]

    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/extraction_{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 初始化LLM客户端
    logger.info("🔧 初始化LLM客户端...")

    # 从配置文件加载配置
    import yaml
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = PROJECT_ROOT / "config" / "extraction_config.yaml"

    logger.info(f"📋 使用配置文件: {config_path}")

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"⚠️ 配置文件不存在: {config_path}")
        config = {}

    # 初始化客户端
    llm_config = config.get('llm_clients', {})

    # 学生模型（文本提取）
    kimi_config = llm_config.get('kimi_client', {})
    kimi_client = build_client(
        kimi_config.get('provider', 'moonshot'),
        kimi_config.get('model_name', 'kimi-k2-0905-preview')
    )

    deepseek_config = llm_config.get('deepseek_client', {})
    deepseek_client = build_client(
        deepseek_config.get('provider', 'deepseek'),
        deepseek_config.get('model_name', 'deepseek-chat')
    )

    glm47_config = llm_config.get('glm47_client', {})
    glm47_client = build_client(
        glm47_config.get('provider', 'zhipuai'),
        glm47_config.get('model_name', 'glm-4.7')
    )

    # 视觉模型（表格提取）
    glm46v_config = llm_config.get('glm46v_client', {})
    glm46v_client = build_client(
        glm46v_config.get('provider', 'zhipuai'),
        glm46v_config.get('model_name', 'glm-4.6v')
    )

    # 教师模型（聚合）
    aggregation_config = llm_config.get('aggregation_client', {})
    aggregation_client = build_client(
        aggregation_config.get('provider', 'openai'),
        aggregation_config.get('model_name', 'gpt-5.1')
    )

    # 兼容旧配置
    text_config = llm_config.get('text_client', {})
    text_client = build_client(
        text_config.get('provider', 'deepseek'),
        text_config.get('model_name', 'deepseek-chat')
    )

    multimodal_config = llm_config.get('multimodal_client', {})
    multimodal_client = build_client(
        multimodal_config.get('provider', 'zhipuai'),
        multimodal_config.get('model_name', 'glm-4.6v')
    )

    review_config = llm_config.get('review_client', {})
    review_client = build_client(
        review_config.get('provider', 'openai'),
        review_config.get('model_name', 'gpt-5.1')
    )

    # 检查是否使用扩展版配置
    is_expanded = "v7_expanded" in str(config_path)

    # 设置prompt路径
    if is_expanded:
        text_prompt = "prompts/prompts_extract_from_text_v7_expanded.txt"
        table_prompt = "prompts/prompts_extract_from_table_v7_expanded.txt"
        logger.info("📋 使用扩展版Prompt (v7.0) - 接受固定化/粗酶")
    else:
        text_prompt = config.get('file_paths', {}).get('prompt_text', 'prompts/prompts_extract_from_text.txt')
        table_prompt = config.get('file_paths', {}).get('prompt_table', 'prompts/prompts_extract_from_table.txt')

    # 初始化pipeline
    pipeline = EnhancedExtractionPipeline(
        use_paper_level_aggregation=True,
        kimi_client=kimi_client,
        deepseek_client=deepseek_client,
        glm47_client=glm47_client,
        glm46v_client=glm46v_client,
        aggregation_client=aggregation_client,
        text_client=text_client,
        multimodal_client=multimodal_client,
        review_client=review_client,
        text_prompt_path=text_prompt,
        table_prompt_path=table_prompt,
        max_workers=max_workers,
        use_full_md=True,
        enable_record_merge=True,
        enable_sequence_enrichment=True,
        require_sequence=not is_expanded,
    )

    # 运行提取
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 开始提取流程")
    logger.info(f"{'='*80}")
    logger.info(f"输入目录: {input_path}")
    logger.info(f"输出目录: {output_path}")
    logger.info(f"论文数量: {len(paper_dirs)}")
    logger.info(f"{'='*80}\n")

    stats = pipeline.run(
        paper_dirs=[str(d) for d in paper_dirs],
        output_dir=str(output_path)
    )

    # 打印统计
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 提取完成统计")
    logger.info(f"{'='*80}")
    statistics = stats['statistics']
    logger.info(f"总论文数: {statistics['total_papers']}")
    logger.info(f"成功处理: {statistics['processed_papers']}")
    logger.info(f"预检查跳过: {statistics.get('skipped_papers', 0)}")
    logger.info(f"处理失败: {statistics['failed_papers']}")
    logger.info(f"总记录数: {statistics['total_records']}")
    logger.info(f"平均时间: {statistics['avg_time_per_paper']:.2f}秒/篇")
    logger.info(f"{'='*80}\n")

    # 保存统计
    stats_file = output_path / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ 统计信息已保存: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description='MycoExtract: 霉菌毒素降解酶数据提取流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
质量评分系统（3级）:
  Score 3 (好): 核心要素 + ≥3重要要素 + ≥1加分项
  Score 2 (中): 核心要素 + ≥1重要要素
  Score 1 (差): 缺少核心要素

示例:
  python scripts/run_extraction.py -i data/papers
  python scripts/run_extraction.py -i data/papers -c config/extraction_config_v7_expanded.yaml
  python scripts/run_extraction.py -i data/papers -n 10  # 测试10篇
        """
    )
    parser.add_argument('--input-dir', '-i', required=True,
                        help='输入目录（包含解析后的论文文件夹）')
    parser.add_argument('--output-dir', '-o',
                        help='输出目录（默认：results/extraction_TIMESTAMP）')
    parser.add_argument('--limit', '-n', type=int,
                        help='限制处理的论文数量（用于测试）')
    parser.add_argument('--config', '-c',
                        help='配置文件路径（默认：config/extraction_config.yaml）')
    parser.add_argument('--max-workers', type=int, default=2,
                        help='最大并发数（默认：2，适配GLM-4.7限制）')

    args = parser.parse_args()

    run_extraction(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        config_file=args.config,
        max_workers=args.max_workers
    )


if __name__ == '__main__':
    main()
