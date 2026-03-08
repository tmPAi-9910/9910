"""
TmpAi Lite - Lightweight LLM with Built-in Tokenizer

A lightweight 1.6B parameter language model with integrated tokenizer
and 4-bit quantization support.

Usage:
    from tmpai_lite import TmpAiLite, TmpAiLiteTokenizer, QuantizationConfig
    
    # Initialize tokenizer
    tokenizer = TmpAiLiteTokenizer()
    
    # Initialize model (normal)
    model = TmpAiLite()
    
    # Or with 4-bit quantization
    config = QuantizationConfig(load_in_4bit=True)
    model = TmpAiLite.from_pretrained("path", quantization_config=config)
"""

__version__ = "1.0.0"
__author__ = "TmpAi-0199"

from tmpai_lite.models import TmpAiLite
from tmpai_lite.tokenizer import TmpAiLiteTokenizer
from tmpai_lite.quantization import QuantizationConfig

__all__ = [
    'TmpAiLite',
    'TmpAiLiteTokenizer',
    'QuantizationConfig',
]
