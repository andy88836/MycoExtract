"""
Agent系统 - 数据处理与智能审核

推荐使用 review_pipeline.py:
- DataProcessor: 数据处理（规则，不需要LLM）
- ReviewerAgent: 智能审核（LLM + 联网工具）
- PostExtractionPipeline: 完整后处理流水线

流程：
    Extract → DataProcessor(规则) → ReviewerAgent(LLM+联网) → HITL
"""

# 后处理流水线（推荐使用）
from .review_pipeline import (
    # 数据结构
    ReviewDecision,
    ProcessingResult,
    ReviewResult,
    
    # 组件
    DataProcessor,           # 数据处理器（规则，不需要LLM）
    WebVerificationTools,    # 联网验证工具
    ReviewerAgent,           # 智能审核Agent（LLM + 工具）
    
    # 主类
    PostExtractionPipeline,
    
    # 便捷函数
    create_pipeline,
    run_post_extraction,
)

__all__ = [
    # 数据结构
    "ReviewDecision",
    "ProcessingResult",
    "ReviewResult",
    
    # 组件
    "DataProcessor",
    "WebVerificationTools",
    "ReviewerAgent",
    
    # 主类
    "PostExtractionPipeline",
    
    # 便捷函数
    "create_pipeline",
    "run_post_extraction",
]
