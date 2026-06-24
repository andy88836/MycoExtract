"""
MycoExtract 基础使用示例

演示如何使用 MycoExtract 进行酶学数据提取
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.enhanced_pipeline import EnhancedExtractionPipeline
from src.llm_clients import build_client
from src.utils.logging_config import setup_logging, get_logger

# 初始化日志
setup_logging()
logger = get_logger(__name__)


def basic_extraction_example():
    """基础提取示例"""
    # 1. 初始化客户端
    kimi_client = build_client("moonshot", "kimi-k2-0905-preview")
    deepseek_client = build_client("deepseek", "deepseek-chat")
    glm47_client = build_client("zhipuai", "glm-4.7")
    glm46v_client = build_client("zhipuai", "glm-4.6v")
    aggregation_client = build_client("openai", "gpt-5.1")

    # 2. 初始化流水线
    pipeline = EnhancedExtractionPipeline(
        use_paper_level_aggregation=True,
        kimi_client=kimi_client,
        deepseek_client=deepseek_client,
        glm47_client=glm47_client,
        glm46v_client=glm46v_client,
        aggregation_client=aggregation_client,
        max_workers=2,
    )

    # 3. 运行提取
    input_dir = "data/papers"
    output_dir = "results/example_extraction"

    stats = pipeline.run(
        paper_dirs=[input_dir],
        output_dir=output_dir
    )

    # 4. 查看结果
    print(f"处理论文数: {stats['statistics']['processed_papers']}")
    print(f"提取记录数: {stats['statistics']['total_records']}")


def custom_config_example():
    """自定义配置示例"""
    import yaml

    # 加载自定义配置
    config_path = "config/extraction_config_v7_expanded.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 使用配置初始化客户端
    llm_config = config['llm_clients']

    pipeline = EnhancedExtractionPipeline(
        use_paper_level_aggregation=True,
        kimi_client=build_client(
            llm_config['kimi_client']['provider'],
            llm_config['kimi_client']['model_name']
        ),
        # ... 其他客户端
        text_prompt_path="prompts/prompts_extract_from_text_v7_expanded.txt",
        table_prompt_path="prompts/prompts_extract_from_table_v7_expanded.txt",
        max_workers=2,
    )

    return pipeline


def batch_processing_example():
    """批处理示例"""
    paper_dirs = [
        "data/papers/paper1",
        "data/papers/paper2",
        "data/papers/paper3",
    ]

    # 初始化流水线 (同上)
    # ...

    # 批量处理
    for paper_dir in paper_dirs:
        try:
            result = pipeline.run([paper_dir], "results/batch")
            logger.info(f"✅ 完成: {paper_dir}")
        except Exception as e:
            logger.error(f"❌ 失败: {paper_dir}, {e}")


def quality_assessment_example():
    """质量评估示例"""
    from src.utils.data_validator import DataValidator

    # 加载提取结果
    import json
    with open("results/extraction/combined_results.json", 'r') as f:
        records = json.load(f)

    # 计算置信度分数
    scores = {}
    for record in records:
        score = DataValidator.calculate_confidence(record)
        scores[record['enzyme_name']] = score

    # 统计分布
    score_counts = {1: 0, 2: 0, 3: 0}
    for score in scores.values():
        score_counts[score] += 1

    print("置信度分布:")
    for score, count in score_counts.items():
        print(f"  {score}分: {count}条记录")


if __name__ == "__main__":
    # 运行基础示例
    basic_extraction_example()
