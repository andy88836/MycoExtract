"""
内容块预过滤器 - 在LLM提取前快速过滤无价值块

目的：避免将参考文献、摘要、致谢等非数据块发送给LLM，节省时间和成本
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ContentFilter:
    """
    智能内容过滤器
    
    快速识别并过滤掉不太可能包含酶动力学数据的文本块：
    - 参考文献列表
    - 摘要/关键词
    - 致谢/作者信息
    - 版权声明
    - 纯数学公式块
    """
    
    # 参考文献特征（标题级关键词）
    REFERENCE_SECTION_PATTERNS = [
        r'^references?\s*$',
        r'^bibliography\s*$',
        r'^literature\s+cited\s*$',
        r'^works\s+cited\s*$',
        r'^citations?\s*$',
    ]
    
    # 参考文献条目特征（内容级特征）
    REFERENCE_ENTRY_PATTERNS = [
        r'doi:\s*10\.\d{4,}',  # DOI链接
        r'https?://doi\.org/',  # DOI URL
        r'\(\d{4}\)\.',  # (2020). 格式
        r'et\s+al\.\s*\(\d{4}\)',  # et al. (2020)
        r'^\s*\[\d+\]',  # [1] [2] 编号
        r'^\s*\d+\.\s+\w+,\s+\w+\..*\(\d{4}\)',  # 1. Author, A. (2020)
    ]
    
    # 其他非数据块特征
    NON_DATA_PATTERNS = [
        r'^abstract\s*$',
        r'^keywords?\s*$',
        r'^acknowledgments?\s*$',
        r'^funding\s*$',
        r'^conflict\s+of\s+interest',
        r'^author\s+contributions?',
        r'^copyright',
        r'^©\s*\d{4}',
        r'^received:.*accepted:',  # 投稿日期
    ]
    
    # 可能包含数据的关键词（正向信号）
    DATA_KEYWORDS = [
        'km', 'vmax', 'kcat', 'ki', 'ic50',
        'enzyme', 'kinetic', 'activity',
        'degradation', 'mycotoxin', 'aflatoxin',
        'substrate', 'product', 'reaction',
        'temperature', 'ph', 'optimal',
        'μm', 'mm', 'nm', 'µm',  # 单位
        'min⁻¹', 's⁻¹', 'h⁻¹',  # 速率单位
    ]
    
    def __init__(
        self,
        min_text_length: int = 100,
        max_reference_ratio: float = 0.7,
        enable_keyword_check: bool = True
    ):
        """
        Args:
            min_text_length: 最小文本长度（太短的块通常无价值）
            max_reference_ratio: 参考文献特征比例阈值
            enable_keyword_check: 是否启用数据关键词检查
        """
        self.min_text_length = min_text_length
        self.max_reference_ratio = max_reference_ratio
        self.enable_keyword_check = enable_keyword_check
        
        # 编译正则表达式
        self.ref_section_regex = [
            re.compile(p, re.IGNORECASE) for p in self.REFERENCE_SECTION_PATTERNS
        ]
        self.ref_entry_regex = [
            re.compile(p, re.IGNORECASE) for p in self.REFERENCE_ENTRY_PATTERNS
        ]
        self.non_data_regex = [
            re.compile(p, re.IGNORECASE) for p in self.NON_DATA_PATTERNS
        ]
        
        logger.info("ContentFilter initialized")
        logger.info(f"  - Min text length: {min_text_length}")
        logger.info(f"  - Max reference ratio: {max_reference_ratio}")
        logger.info(f"  - Keyword check: {enable_keyword_check}")
    
    def filter_text_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤文本块列表
        
        Args:
            blocks: 原始文本块列表
            
        Returns:
            过滤后的文本块列表
        """
        if not blocks:
            return blocks
        
        filtered = []
        stats = {
            'total': len(blocks),
            'too_short': 0,
            'reference_section': 0,
            'reference_entries': 0,
            'non_data_section': 0,
            'no_keywords': 0,
            'kept': 0
        }
        
        for i, block in enumerate(blocks, 1):
            text = self._get_text_from_block(block)
            
            if not text:
                continue
            
            # 检查1: 长度过短
            if len(text) < self.min_text_length:
                stats['too_short'] += 1
                continue
            
            # 检查2: 参考文献章节标题
            if self._is_reference_section_title(text):
                stats['reference_section'] += 1
                logger.debug(f"  Block {i}: Skipped (reference section title)")
                continue
            
            # 检查3: 参考文献条目
            if self._is_reference_entries(text):
                stats['reference_entries'] += 1
                logger.debug(f"  Block {i}: Skipped (reference entries)")
                continue
            
            # 检查4: 其他非数据块
            if self._is_non_data_section(text):
                stats['non_data_section'] += 1
                logger.debug(f"  Block {i}: Skipped (non-data section)")
                continue
            
            # 检查5: 关键词检查（可选）
            if self.enable_keyword_check:
                if not self._has_data_keywords(text):
                    stats['no_keywords'] += 1
                    logger.debug(f"  Block {i}: Skipped (no data keywords)")
                    continue
            
            # 通过所有检查
            filtered.append(block)
            stats['kept'] += 1
        
        # 打印统计
        logger.info(f"📊 Content filtering results:")
        logger.info(f"  Total blocks: {stats['total']}")
        logger.info(f"  ✅ Kept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
        logger.info(f"  ❌ Filtered: {stats['total'] - stats['kept']}")
        logger.info(f"     - Too short: {stats['too_short']}")
        logger.info(f"     - Reference sections: {stats['reference_section']}")
        logger.info(f"     - Reference entries: {stats['reference_entries']}")
        logger.info(f"     - Non-data sections: {stats['non_data_section']}")
        if self.enable_keyword_check:
            logger.info(f"     - No keywords: {stats['no_keywords']}")
        
        return filtered
    
    def _get_text_from_block(self, block: Dict[str, Any]) -> str:
        """从块中提取文本内容"""
        if isinstance(block, str):
            return block
        
        # 尝试多个可能的字段名
        for field in ['text', 'content', 'body', 'para']:
            if field in block:
                text = block[field]
                if isinstance(text, str):
                    return text.strip()
                elif isinstance(text, list):
                    return ' '.join(str(t) for t in text).strip()
        
        return ''
    
    def _is_reference_section_title(self, text: str) -> bool:
        """检查是否是参考文献章节标题"""
        # 只检查前50个字符（标题通常很短）
        title_part = text[:50].strip()
        
        for regex in self.ref_section_regex:
            if regex.match(title_part):
                return True
        
        return False
    
    def _is_reference_entries(self, text: str) -> bool:
        """检查是否是参考文献条目列表"""
        # 计算匹配参考文献特征的行数比例
        lines = text.split('\n')
        if len(lines) < 3:
            return False
        
        match_count = 0
        for line in lines:
            if not line.strip():
                continue
            
            for regex in self.ref_entry_regex:
                if regex.search(line):
                    match_count += 1
                    break
        
        # 如果超过阈值的行匹配参考文献特征，判定为参考文献块
        ratio = match_count / len(lines)
        return ratio > self.max_reference_ratio
    
    def _is_non_data_section(self, text: str) -> bool:
        """检查是否是其他非数据章节"""
        # 检查前100个字符
        header = text[:100].strip().lower()
        
        for regex in self.non_data_regex:
            if regex.search(header):
                return True
        
        return False
    
    def _has_data_keywords(self, text: str) -> bool:
        """检查是否包含数据相关关键词"""
        text_lower = text.lower()
        
        # 只要包含任意一个数据关键词就保留
        for keyword in self.DATA_KEYWORDS:
            if keyword in text_lower:
                return True
        
        return False


def create_default_filter() -> ContentFilter:
    """创建默认配置的过滤器"""
    return ContentFilter(
        min_text_length=100,
        max_reference_ratio=0.7,
        enable_keyword_check=True
    )


if __name__ == "__main__":
    """测试过滤器"""
    logging.basicConfig(level=logging.DEBUG)
    
    # 测试文本
    test_blocks = [
        {"text": "Too short"},  # 应被过滤
        {"text": "REFERENCES\n\nHere are the references..."},  # 应被过滤
        {"text": "1. Smith, J. et al. (2020). doi:10.1234/test"},  # 应被过滤
        {"text": "The enzyme showed a Km of 0.5 μM and Vmax of 100 nmol/min. This kinetic data suggests..."},  # 应保留
        {"text": "ACKNOWLEDGMENTS\n\nWe thank..."},  # 应被过滤
    ]
    
    filter = create_default_filter()
    filtered = filter.filter_text_blocks(test_blocks)
    
    print(f"\n✅ Kept {len(filtered)} / {len(test_blocks)} blocks")
