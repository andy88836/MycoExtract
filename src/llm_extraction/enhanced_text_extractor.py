"""
增强版文本提取器 - 支持上下文重叠分块

主要改进：
1. 智能分块时保留上下文重叠
2. 避免跨块信息丢失
"""

import re
import logging
from typing import List, Dict, Any
from src.llm_extraction.text_extractor import TextExtractor

logger = logging.getLogger(__name__)


class EnhancedTextExtractor(TextExtractor):
    """
    增强版文本提取器
    
    新增功能：
    - 分块时保留上下文重叠（overlap）
    - 提升跨块信息关联准确性
    """
    
    def __init__(
        self,
        llm_client,
        prompt_path: str,
        context_overlap_sentences: int = 2,
        max_chunk_size: int = 6000
    ):
        """
        Args:
            llm_client: LLM客户端
            prompt_path: 提示词文件路径
            context_overlap_sentences: 分块时保留的上下文句子数
            max_chunk_size: 最大分块大小（字符数）
        """
        super().__init__(llm_client, prompt_path)
        self.context_overlap_sentences = context_overlap_sentences
        self.max_chunk_size = max_chunk_size
        
        logger.info(f"✓ EnhancedTextExtractor initialized")
        logger.info(f"  - Context overlap: {context_overlap_sentences} sentences")
        logger.info(f"  - Max chunk size: {max_chunk_size} chars")
    
    def _merge_text_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        重写分块方法：添加上下文重叠
        
        策略：
        1. 合并连续文本块
        2. 当累积文本超过max_chunk_size时，在句子边界分割
        3. **关键改进**：下一个块开头包含上一个块的最后N句话
        
        Args:
            blocks: 原始文本blocks
            
        Returns:
            带上下文重叠的merged blocks
        """
        merged = []
        current_text = []
        current_length = 0
        previous_overlap = []  # 保存上一个块的overlap内容
        
        def flush_current_text():
            """将累积的文本分割并添加overlap"""
            if not current_text:
                return
            
            full_text = "\n\n".join(current_text)
            
            # 如果有previous_overlap，添加到开头
            if previous_overlap:
                overlap_text = " ".join(previous_overlap)
                full_text = f"[CONTEXT FROM PREVIOUS CHUNK]: {overlap_text}\n\n{full_text}"
                logger.debug(f"  Added {len(previous_overlap)} sentences overlap")
            
            # 检查是否需要分割
            if len(full_text) <= self.max_chunk_size:
                merged.append({
                    "type": "text",
                    "content": full_text
                })
            else:
                # 需要分割：在句子边界分割，并保留overlap
                chunks = self._split_with_overlap(full_text)
                for chunk in chunks:
                    merged.append({
                        "type": "text",
                        "content": chunk
                    })
            
            current_text.clear()
            nonlocal current_length
            current_length = 0
        
        # 遍历所有blocks
        for block in blocks:
            if block.get("type") == "text":
                text = block.get("text", "").strip()
                
                if not text or len(text) < 50:
                    continue
                
                # 检查是否需要flush
                if current_length + len(text) > self.max_chunk_size and current_text:
                    # 提取overlap
                    previous_overlap = self._extract_last_sentences(
                        "\n\n".join(current_text),
                        self.context_overlap_sentences
                    )
                    flush_current_text()
                
                current_text.append(text)
                current_length += len(text)
        
        # 处理剩余文本
        if current_text:
            flush_current_text()
        
        logger.info(f"  Merged into {len(merged)} chunks with context overlap")
        
        return merged
    
    def _split_with_overlap(self, text: str) -> List[str]:
        """
        将长文本在句子边界分割，并在chunk之间添加overlap
        
        Args:
            text: 待分割文本
            
        Returns:
            分割后的文本chunks列表
        """
        # 按句子分割（支持中英文）
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_buffer = []  # 上一个chunk的overlap
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # 检查是否需要开始新chunk
            if current_length + sentence_len > self.max_chunk_size and current_chunk:
                # 保存当前chunk（带上一个的overlap）
                chunk_text = " ".join(current_chunk)
                if overlap_buffer:
                    overlap_text = " ".join(overlap_buffer)
                    chunk_text = f"[CONTEXT]: {overlap_text}\n\n{chunk_text}"
                
                chunks.append(chunk_text)
                
                # 提取overlap用于下一个chunk
                overlap_buffer = current_chunk[-self.context_overlap_sentences:]
                
                # 开始新chunk
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # 处理最后一个chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if overlap_buffer:
                overlap_text = " ".join(overlap_buffer)
                chunk_text = f"[CONTEXT]: {overlap_text}\n\n{chunk_text}"
            chunks.append(chunk_text)
        
        logger.info(f"  ✂️ Split long text ({len(text)} chars) into {len(chunks)} overlapping chunks")
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子（支持中英文）
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 句子分割正则（支持中英文标点）
        sentence_pattern = r'[^.!?。！？]+[.!?。！？]+'
        sentences = re.findall(sentence_pattern, text)
        
        # 如果正则没匹配到（可能没有标点），按换行分割
        if not sentences:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
        return sentences
    
    def _extract_last_sentences(self, text: str, n: int) -> List[str]:
        """
        提取文本的最后N句话
        
        Args:
            text: 输入文本
            n: 句子数量
            
        Returns:
            最后N句话列表
        """
        sentences = self._split_into_sentences(text)
        return sentences[-n:] if len(sentences) >= n else sentences


# 测试代码
if __name__ == "__main__":
    # 测试overlap功能
    test_text = """
    Enzyme CotA is a laccase from Bacillus subtilis. It shows high activity towards AFB1.
    The optimal pH is 7.0 and the optimal temperature is 45°C.
    Kinetic studies revealed a Km of 0.5 μM and Vmax of 100 μM/min.
    The enzyme degrades AFB1 to AFQ1 with 85% efficiency in 24 hours.
    Product analysis confirmed the formation of less toxic metabolites.
    """ * 50  # 重复50次模拟长文本
    
    print(f"Original text length: {len(test_text)} chars")
    print(f"Contains approximately {len(test_text.split('.'))} sentences")
    
    # 创建extractor（需要mock llm_client）
    class MockClient:
        def chat(self, *args, **kwargs):
            return "[]"
    
    extractor = EnhancedTextExtractor(
        llm_client=MockClient(),
        prompt_path="dummy.txt",
        context_overlap_sentences=2,
        max_chunk_size=500
    )
    
    # 测试分割
    chunks = extractor._split_with_overlap(test_text)
    
    print(f"\n✓ Split into {len(chunks)} chunks with overlap")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} (first 200 chars) ---")
        print(chunk[:200])
        print("...")
