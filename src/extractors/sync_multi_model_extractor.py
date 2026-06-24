"""
多模型投票提取器的同步包装器

由于 enhanced_pipeline.py 使用同步代码，此包装器提供同步接口调用异步的 MultiModelExtractor。
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from src.extractors.multi_model_extractor import MultiModelExtractor

logger = logging.getLogger(__name__)


class SyncMultiModelExtractor:
    """
    MultiModelExtractor的同步包装器
    
    提供与现有单模型提取器兼容的同步接口。
    """
    
    def __init__(self, multi_model_extractor: MultiModelExtractor):
        """
        Args:
            multi_model_extractor: 异步的MultiModelExtractor实例
        """
        self.extractor = multi_model_extractor
        self._loop = None
    
    def _get_loop(self):
        """获取或创建事件循环"""
        if self._loop is None or self._loop.is_closed():
            try:
                # 尝试获取当前事件循环
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                # 如果没有事件循环，创建一个新的
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def extract_from_text(
        self,
        text_content: str,
        prompt_template: str,
        doi: str,
        block_id: int
    ) -> List[Dict[str, Any]]:
        """
        同步接口：从文本提取（3模型投票）
        
        Args:
            text_content: 文本内容
            prompt_template: 提示词模板
            doi: 论文DOI
            block_id: 块ID
            
        Returns:
            提取的记录列表（包含投票信息）
        """
        loop = self._get_loop()
        
        # 运行异步方法
        return loop.run_until_complete(
            self.extractor.extract_from_text(
                text_content=text_content,
                prompt_template=prompt_template,
                doi=doi,
                block_id=block_id
            )
        )
    
    def extract_from_table(
        self,
        table_content: str,
        prompt_template: str,
        doi: str,
        block_id: int
    ) -> List[Dict[str, Any]]:
        """
        同步接口：从表格提取（3模型投票）
        
        Args:
            table_content: 表格HTML内容
            prompt_template: 提示词模板
            doi: 论文DOI
            block_id: 块ID
            
        Returns:
            提取的记录列表（包含投票信息）
        """
        loop = self._get_loop()
        
        return loop.run_until_complete(
            self.extractor.extract_from_table(
                table_content=table_content,
                prompt_template=prompt_template,
                doi=doi,
                block_id=block_id
            )
        )
    
    def extract_from_figure(
        self,
        image_path: str,
        prompt_template: str,
        doi: str,
        block_id: int
    ) -> List[Dict[str, Any]]:
        """
        同步接口：从图片提取（单模型，无投票）
        
        Args:
            image_path: 图片路径
            prompt_template: 提示词模板
            doi: 论文DOI
            block_id: 块ID
            
        Returns:
            提取的记录列表
        """
        loop = self._get_loop()
        
        return loop.run_until_complete(
            self.extractor.extract_from_figure(
                image_path=image_path,
                prompt_template=prompt_template,
                doi=doi,
                block_id=block_id
            )
        )


def create_sync_multi_model_extractor(
    kimi_client,
    deepseek_client,
    glm46_client,
    glm46v_client: Optional[Any] = None
) -> SyncMultiModelExtractor:
    """
    创建同步的多模型提取器
    
    Args:
        kimi_client: Kimi LLM客户端
        deepseek_client: DeepSeek LLM客户端
        glm46_client: GLM-4.6 LLM客户端
        glm46v_client: GLM-4.6V多模态客户端（可选）
        
    Returns:
        同步多模型提取器实例
    """
    # 创建异步提取器
    async_extractor = MultiModelExtractor(
        kimi_client=kimi_client,
        deepseek_client=deepseek_client,
        glm46_client=glm46_client,
        glm46v_client=glm46v_client
    )
    
    # 包装为同步接口
    return SyncMultiModelExtractor(async_extractor)
