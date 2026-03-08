"""
TmpAi Lite Quantization Module

4-bit quantization support using bitsandbytes.
"""

from tmpai_lite.quantization.quantizer import QuantizationConfig, apply_4bit_quantization

__all__ = ['QuantizationConfig', 'apply_4bit_quantization']
