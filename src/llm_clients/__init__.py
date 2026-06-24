"""
LLM Client Module - Simplified Version

This module provides a unified interface for interacting with 3 core LLM providers:
- OpenAI (GPT-4o, GPT-4o-mini)
- DeepSeek (deepseek-chat, deepseek-reasoner)
- ZhipuAI (GLM-4.5V, GLM-4V-Plus)
"""

from .providers import (
    BaseLLMClient,
    OpenAIClient,
    DeepSeekClient,
    ZhipuAIClient,
    build_client
)

__all__ = [
    'BaseLLMClient',
    'OpenAIClient',
    'DeepSeekClient',
    'ZhipuAIClient',
    'build_client'
]

__version__ = '2.0.0'  # Simplified version