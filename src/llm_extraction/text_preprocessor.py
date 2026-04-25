"""
文本预处理模块
用于在 LLM 提取前清洗和规范化文本，提升提取成功率

主要功能：
1. **去除 References 部分** (最重要！避免浪费 token 和提取无用信息)
2. LaTeX 格式规范化（$( 0 . 3 3 ~ \mu \mathrm { M } )$ → 0.33 μM）
3. 清除多余空格和噪音
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """文本预处理器"""
    
    # LaTeX 单位映射表 (保留：对提取准确性至关重要)
    LATEX_UNITS = {
        r'\\mu\\mathrm\s*\{\s*M\s*\}': 'μM',
        r'\\mu\s*M': 'μM',
        r'\\mathrm\s*\{\s*m\s*M\s*\}': 'mM',
        r'\\mathrm\s*\{\s*M\s*\}': 'M',
        r'\\mathrm\s*\{\s*s\s*\}': 's',
        r'\\mathrm\s*\{\s*min\s*\}': 'min',
        r'\\mathrm\s*\{\s*h\s*\}': 'h',
        r'\^\s*\{\s*-\s*1\s*\}': '⁻¹',
        r'\^\s*\{\s*\\circ\s*\}': '°',
        r'\\mathrm\s*\{\s*~\s*C\s*~\s*\}': '°C',
        r'\\mathrm\s*\{\s*p\s*H\s*~\s*\}': 'pH ',
    }
    
    # References 部分的常见标题模式
    REFERENCE_HEADERS = [
        r'\n\s*REFERENCES?\s*\n',
        r'\n\s*References?\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
        r'\n\s*Literature Cited\s*\n',
        r'\n\s*Works Cited\s*\n',
        # 带编号的情况
        r'\n\s*\d+\.?\s*REFERENCES?\s*\n',
        r'\n\s*\d+\.?\s*References?\s*\n',
    ]
    
    # Introduction 部分的常见标题模式
    INTRODUCTION_HEADERS = [
        r'\n\s*INTRODUCTION\s*\n',
        r'\n\s*Introduction\s*\n',
        r'\n\s*1\.?\s*INTRODUCTION\s*\n',
        r'\n\s*1\.?\s*Introduction\s*\n',
        r'\n\s*I\.?\s*INTRODUCTION\s*\n',
        r'\n\s*I\.?\s*Introduction\s*\n',
    ]
    
    # Introduction 结束的标志 (Methods/Results/Materials 等章节开始)
    INTRODUCTION_END_MARKERS = [
        r'\n\s*(?:MATERIALS?\s+AND\s+)?METHODS?\s*\n',
        r'\n\s*(?:Materials?\s+and\s+)?Methods?\s*\n',
        r'\n\s*RESULTS?\s*\n',
        r'\n\s*Results?\s*\n',
        r'\n\s*EXPERIMENTAL\s*\n',
        r'\n\s*Experimental\s*\n',
        r'\n\s*\d+\.?\s*(?:MATERIALS?\s+AND\s+)?METHODS?\s*\n',
        r'\n\s*\d+\.?\s*(?:Materials?\s+and\s+)?Methods?\s*\n',
        r'\n\s*\d+\.?\s*RESULTS?\s*\n',
        r'\n\s*\d+\.?\s*Results?\s*\n',
    ]
    
    def remove_introduction(self, text: str) -> str:
        """
        去除文献的 Introduction 部分
        
        原因：Introduction 通常是背景介绍和文献综述，不包含具体的实验数据和动力学参数
        
        策略：
        1. 查找 "Introduction" 标题
        2. 查找 Introduction 结束的标志（如 "Methods", "Results" 等章节）
        3. 删除 Introduction 到下一章节之间的内容
        
        Examples:
            "INTRODUCTION\n背景介绍...\n\nMETHODS\n实验方法..." 
            → "METHODS\n实验方法..."
        """
        # 查找 Introduction 开始位置
        intro_start = None
        intro_pattern_used = None
        
        for pattern in self.INTRODUCTION_HEADERS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                intro_start = match.start()
                intro_pattern_used = pattern
                break
        
        if intro_start is None:
            # 没找到 Introduction，返回原文
            return text
        
        # 查找 Introduction 结束位置（下一个章节开始）
        intro_end = None
        text_after_intro = text[intro_start:]
        
        for pattern in self.INTRODUCTION_END_MARKERS:
            match = re.search(pattern, text_after_intro, re.IGNORECASE)
            if match:
                # 找到下一章节，保留该章节标题
                intro_end = intro_start + match.start()
                break
        
        if intro_end is None:
            # 没找到结束标志，可能 Introduction 后面就是 References 或文末
            # 保守策略：不删除（避免误删有用内容）
            logger.debug(f"  ⚠️  Found Introduction but no ending marker, keeping it")
            return text
        
        # 删除 Introduction 部分
        text_result = text[:intro_start] + text[intro_end:]
        removed_chars = len(text) - len(text_result)
        logger.debug(f"  🗑️  Removed Introduction section: {removed_chars} chars")
        
        return text_result
    
    def remove_references(self, text: str) -> str:
        """
        去除文献的 References 部分
        
        原因：References 部分通常占 20-30% 的文本，但不包含任何动力学数据，
        会浪费大量 tokens 并可能误导 LLM 提取引用中的数据
        
        策略：
        1. 查找 "References" 标题
        2. 删除该标题及其后的所有内容
        3. 如果没找到标题，尝试检测引用列表模式（连续的编号文献）
        
        Examples:
            "...result shows that AFO... \n\nREFERENCES\n1. Smith et al..." 
            → "...result shows that AFO..."
        """
        # 策略 1: 查找明确的 References 标题
        for pattern in self.REFERENCE_HEADERS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # 找到标题，截断到这里
                text_before_refs = text[:match.start()]
                logger.debug(f"  🗑️  Removed References section: {len(text) - len(text_before_refs)} chars")
                return text_before_refs.rstrip()
        
        # 策略 2: 检测引用列表模式（连续 3 个以上的编号文献）
        # 模式：\n1. Author... \n2. Author... \n3. Author...
        # 这种情况通常出现在 PDF 解析时 References 标题丢失
        lines = text.split('\n')
        reference_start = None
        consecutive_citations = 0
        
        for i, line in enumerate(lines):
            # 检测引用格式：数字开头 + 作者名 + 年份
            # 例如: "1. Smith, J. (2020)..." 或 "[1] Jones et al., 2019..."
            if re.match(r'^\s*[\[\(]?\d+[\]\)\.]\s+[A-Z][a-z]+', line.strip()):
                consecutive_citations += 1
                if consecutive_citations == 1:
                    reference_start = i
                if consecutive_citations >= 3:
                    # 找到连续 3 个引用，认为这是 References 部分
                    text_before_refs = '\n'.join(lines[:reference_start])
                    logger.debug(f"  🗑️  Detected reference list without header, removed: {len(text) - len(text_before_refs)} chars")
                    return text_before_refs.rstrip()
            else:
                # 重置计数
                consecutive_citations = 0
                reference_start = None
        
        # 没有找到 References，返回原文
        return text
    
    def normalize_latex(self, text: str) -> str:
        """
        规范化 LaTeX 格式为易于 LLM 识别的纯文本
        
        Examples:
            $( 0 . 3 3 ~ \mu \mathrm { M } )$ → 0.33 μM
            $3 9 . 3 ~ \mu \mathrm { M }$ → 39.3 μM
            $\mathrm { p H } ~ 6 . 0$ → pH 6.0
        """
        # 清除波浪号（LaTeX 中的空格）
        text = re.sub(r'\s*~\s*', ' ', text)
        
        # 清除 LaTeX 标记（在处理单位前）
        text = re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', text)  # \mathrm{text} → text
        
        # 替换 LaTeX 单位为纯文本单位
        for latex_pattern, plain_unit in self.LATEX_UNITS.items():
            text = re.sub(latex_pattern, plain_unit, text, flags=re.IGNORECASE)
        
        # 处理 \mu 单独出现的情况
        text = re.sub(r'\\mu\s+', 'μ', text)
        text = re.sub(r'\\mu([A-Z])', r'μ\1', text)  # \muM → μM
        
        # 处理 LaTeX 数学模式：$( value unit )$ → value unit
        # 先提取出数字和单位，然后清理数字中的空格
        def replace_latex_math(match):
            content = match.group(1)
            # 清理数字中的空格：0 . 3 3 → 0.33
            content = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', content)  # 0 . 3 → 0.3
            content = re.sub(r'(\d)\s+(\d)', r'\1\2', content)  # 3 3 → 33
            return content
        
        text = re.sub(r'\$\s*\(\s*([^)]+)\s*\)\s*\$', replace_latex_math, text)
        text = re.sub(r'\$\s*([^$]+)\s*\$', replace_latex_math, text)
        
        # 清除剩余的 LaTeX 命令和特殊字符
        text = re.sub(r'\\[a-zA-Z]+\s*', '', text)  # 清除所有 LaTeX 命令
        text = re.sub(r'[\$\{\}]', '', text)  # 清除剩余的 ${}
        
        return text
    
    def clean_whitespace(self, text: str) -> str:
        """
        清理多余空格，保持文本可读性
        """
        # 清除多余空格（保留段落分隔的双换行）
        text = re.sub(r' +', ' ', text)  # 多个空格 → 单个空格
        text = re.sub(r'\n{3,}', '\n\n', text)  # 多个换行 → 双换行
        text = re.sub(r' \n', '\n', text)  # 行尾空格
        text = re.sub(r'\n ', '\n', text)  # 行首空格
        
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        """
        完整的预处理流程
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text:
            return text
        
        # 1. 保留 Introduction 部分（用户要求保留）
        # text = self.remove_introduction(text)
        
        # 2. 去除 References 部分（最重要！）
        text = self.remove_references(text)
        
        # 3. 规范化 LaTeX
        text = self.normalize_latex(text)
        
        # 4. 清理空格
        text = self.clean_whitespace(text)
        
        return text


# 单例实例
preprocessor = TextPreprocessor()


# 便捷函数
def preprocess_text(text: str) -> str:
    """便捷的文本预处理函数"""
    return preprocessor.preprocess(text)


if __name__ == '__main__':
    # 测试样例 1: LaTeX 规范化
    test_latex = """
    AFO from the fungus A. tabescens can hydrolyze a furan ring of the molecule AFB1.
    The extremely low Km value $( 0 . 3 3 ~ \mu \mathrm { M } )$ determined for this enzyme 
    indicates the high specificity.
    """
    
    print("="*80)
    print("测试 1: LaTeX 规范化")
    print("="*80)
    print("原始文本:")
    print(test_latex)
    print("\n预处理后:")
    print(preprocess_text(test_latex))
    
    # 测试样例 2: 去除 Introduction
    test_with_intro = """
    INTRODUCTION
    
    Mycotoxins are toxic secondary metabolites produced by fungi. Aflatoxin B1 (AFB1) 
    is one of the most potent carcinogens. Previous studies have shown that enzymatic 
    degradation is a promising approach for mycotoxin detoxification.
    
    MATERIALS AND METHODS
    
    The enzyme AFO was purified from Armillaria tabescens. Kinetic parameters were 
    determined using spectrophotometry. The Km value was 0.33 μM and kcat was 63 min⁻¹.
    
    RESULTS
    
    AFO showed high substrate specificity with degradation efficiency of 95%.
    """
    
    print("\n" + "="*80)
    print("测试 2: 去除 Introduction")
    print("="*80)
    print("原始文本:")
    print(test_with_intro)
    print("\n预处理后:")
    result = preprocess_text(test_with_intro)
    print(result)
    print(f"\n字符数: {len(test_with_intro)} → {len(result)} (减少 {len(test_with_intro) - len(result)} 字符)")
    
    # 测试样例 3: 去除 References
    test_with_refs = """
    The enzyme showed high activity with Km = 0.33 μM and kcat = 63 min⁻¹.
    These results suggest potential applications in mycotoxin detoxification.
    
    REFERENCES
    1. Smith, J. et al. (2020) Nature, 123, 456-789.
    2. Jones, A. B. (2019) Science, 234, 567-890.
    3. Brown, C. D. et al. (2018) Cell, 345, 678-901.
    """
    
    print("\n" + "="*80)
    print("测试 3: 去除 References")
    print("="*80)
    print("原始文本:")
    print(test_with_refs)
    print("\n预处理后:")
    print(preprocess_text(test_with_refs))
    
    # 测试样例 4: 同时去除 Introduction 和 References
    test_full = """
    INTRODUCTION
    
    Background information about mycotoxins and their impact on food safety.
    Literature review of enzymatic degradation methods.
    
    METHODS
    
    Enzyme purification and characterization were performed.
    Kinetic assays measured Km = 0.33 μM for AFB1 degradation.
    
    RESULTS
    
    The enzyme AFO showed high catalytic efficiency with kcat/Km = 191 μM⁻¹min⁻¹.
    
    REFERENCES
    1. Author et al. (2020) Journal.
    2. Another Author (2019) Journal.
    """
    
    print("\n" + "="*80)
    print("测试 4: 同时去除 Introduction 和 References")
    print("="*80)
    print("原始文本:")
    print(test_full)
    print("\n预处理后:")
    result = preprocess_text(test_full)
    print(result)
    print(f"\n字符数: {len(test_full)} → {len(result)} (减少 {len(test_full) - len(result)} 字符, {(1 - len(result)/len(test_full))*100:.1f}%)")

