"""
LLM Extraction Module

This module provides specialized extractors for different content types:
- Text blocks and equations
- Tables (with multimodal fusion and verification)
- Figures and charts

All extractors output structured JSON fragments following a unified schema.
"""

from .text_extractor import TextExtractor
from .table_extractor import TableExtractor
from .figure_extractor import FigureExtractor
from .postprocessor import DataPostprocessor

__all__ = [
    'TextExtractor',
    'TableExtractor',
    'FigureExtractor',
    'DataPostprocessor'
]

__version__ = '1.0.0'