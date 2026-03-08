"""
TmpAi Lite Quantizer

4-bit quantization implementation using NF4 format.
Compatible with bitsandbytes-style quantization.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Any
import math


@dataclass
class QuantizationConfig:
    """Configuration for 4-bit quantization."""
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.bnb_4bit_quant_type not in ["nf4", "fp4"]:
            raise ValueError(f"Invalid quant_type: {self.bnb_4bit_quant_type}. Use 'nf4' or 'fp4'.")
        if self.bnb_4bit_compute_dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(f"Invalid compute_dtype: {self.bnb_4bit_compute_dtype}")


class NF4Quantizer:
    """
    NF4 (Normal Float 4) Quantization.
    
    NF4 is a 4-bit quantization format optimized for normally distributed weights.
    It provides better quantization error than uniform quantization for neural network weights.
    """
    
    # NF4 quantization levels (precomputed for normal distribution)
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])
    
    @classmethod
    def quantize(cls, tensor: torch.Tensor, block_size: int = 64) -> Dict[str, Any]:
        """
        Quantize a tensor to NF4 format.
        
        Args:
            tensor: Input tensor to quantize
            block_size: Block size for grouping (default: 64)
        
        Returns:
            Dictionary containing quantized data, scales, and metadata
        """
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Flatten tensor
        tensor = tensor.float()
        
        # Pad to multiple of block_size
        numel = tensor.numel()
        pad_size = (block_size - numel % block_size) % block_size
        if pad_size > 0:
            tensor = torch.cat([tensor.flatten(), torch.zeros(pad_size, device=tensor.device)])
        
        # Reshape into blocks
        tensor = tensor.view(-1, block_size)
        
        # Compute absmax for each block
        absmax = tensor.abs().max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        absmax = torch.clamp(absmax, min=1e-8)
        
        # Normalize to [-1, 1]
        normalized = tensor / absmax
        
        # Quantize to NF4 levels
        # Find nearest NF4 level for each value
        levels = cls.NF4_LEVELS.to(tensor.device)
        distances = torch.abs(normalized.unsqueeze(-1) - levels)
        quantized_indices = distances.argmin(dim=-1).to(torch.uint8)
        
        # Pack two 4-bit values into one byte
        num_blocks, block_len = quantized_indices.shape
        if block_len % 2 == 1:
            # Pad if odd
            quantized_indices = torch.cat([
                quantized_indices,
                torch.zeros(num_blocks, 1, dtype=torch.uint8, device=tensor.device)
            ], dim=1)
            block_len += 1
        
        # Pack: high 4 bits = first value, low 4 bits = second value
        packed = (quantized_indices[:, 0::2] << 4) | quantized_indices[:, 1::2]
        
        return {
            'quantized': packed,
            'absmax': absmax.squeeze(),
            'original_shape': original_shape,
            'original_dtype': original_dtype,
            'block_size': block_size,
            'numel': numel
        }
    
    @classmethod
    def dequantize(cls, quantized_data: Dict[str, Any]) -> torch.Tensor:
        """
        Dequantize NF4 data back to float.
        
        Args:
            quantized_data: Dictionary from quantize()
        
        Returns:
            Dequantized tensor
        """
        packed = quantized_data['quantized']
        absmax = quantized_data['absmax']
        original_shape = quantized_data['original_shape']
        original_dtype = quantized_data['original_dtype']
        block_size = quantized_data['block_size']
        numel = quantized_data['numel']
        
        # Unpack 4-bit values
        high_4bits = (packed >> 4) & 0x0F
        low_4bits = packed & 0x0F
        
        # Interleave
        quantized_indices = torch.stack([high_4bits, low_4bits], dim=2).view(packed.shape[0], -1)
        
        # Get NF4 values
        levels = cls.NF4_LEVELS.to(packed.device)
        dequantized = levels[quantized_indices.long()]
        
        # Apply absmax scaling
        dequantized = dequantized * absmax.unsqueeze(1)
        
        # Flatten and remove padding
        dequantized = dequantized.flatten()[:numel]
        
        # Reshape
        dequantized = dequantized.view(original_shape)
        
        return dequantized.to(original_dtype)


class FP4Quantizer:
    """
    FP4 (Float 4) Quantization.
    
    Simple 4-bit floating point quantization.
    """
    
    FP4_EXPONENT_BITS = 2
    FP4_MANTISSA_BITS = 1
    
    @classmethod
    def quantize(cls, tensor: torch.Tensor, block_size: int = 64) -> Dict[str, Any]:
        """Quantize to FP4 format."""
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        tensor = tensor.float()
        numel = tensor.numel()
        
        # Pad to multiple of block_size
        pad_size = (block_size - numel % block_size) % block_size
        if pad_size > 0:
            tensor = torch.cat([tensor.flatten(), torch.zeros(pad_size, device=tensor.device)])
        
        tensor = tensor.view(-1, block_size)
        absmax = tensor.abs().max(dim=1, keepdim=True)[0]
        absmax = torch.clamp(absmax, min=1e-8)
        
        # Normalize
        normalized = tensor / absmax
        
        # Simple linear quantization to 16 levels
        # Map [-1, 1] to [0, 15]
        quantized = ((normalized + 1) * 7.5).clamp(0, 15).round().to(torch.uint8)
        
        # Pack two 4-bit values
        num_blocks, block_len = quantized.shape
        if block_len % 2 == 1:
            quantized = torch.cat([
                quantized,
                torch.zeros(num_blocks, 1, dtype=torch.uint8, device=tensor.device)
            ], dim=1)
        
        packed = (quantized[:, 0::2] << 4) | quantized[:, 1::2]
        
        return {
            'quantized': packed,
            'absmax': absmax.squeeze(),
            'original_shape': original_shape,
            'original_dtype': original_dtype,
            'block_size': block_size,
            'numel': numel
        }
    
    @classmethod
    def dequantize(cls, quantized_data: Dict[str, Any]) -> torch.Tensor:
        """Dequantize FP4 data."""
        packed = quantized_data['quantized']
        absmax = quantized_data['absmax']
        original_shape = quantized_data['original_shape']
        original_dtype = quantized_data['original_dtype']
        numel = quantized_data['numel']
        
        # Unpack
        high_4bits = (packed >> 4) & 0x0F
        low_4bits = packed & 0x0F
        quantized = torch.stack([high_4bits, low_4bits], dim=2).view(packed.shape[0], -1)
        
        # Dequantize: [0, 15] -> [-1, 1]
        dequantized = (quantized.float() / 7.5) - 1
        
        # Apply absmax
        dequantized = dequantized * absmax.unsqueeze(1)
        dequantized = dequantized.flatten()[:numel]
        dequantized = dequantized.view(original_shape)
        
        return dequantized.to(original_dtype)


def quantize_linear_weight(weight: torch.Tensor, quant_type: str = "nf4") -> Dict[str, Any]:
    """
    Quantize a linear layer weight.
    
    Args:
        weight: Weight tensor
        quant_type: 'nf4' or 'fp4'
    
    Returns:
        Quantized weight data
    """
    if quant_type == "nf4":
        return NF4Quantizer.quantize(weight)
    elif quant_type == "fp4":
        return FP4Quantizer.quantize(weight)
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")


def dequantize_linear_weight(quantized_data: Dict[str, Any], quant_type: str = "nf4") -> torch.Tensor:
    """Dequantize a linear layer weight."""
    if quant_type == "nf4":
        return NF4Quantizer.dequantize(quantized_data)
    elif quant_type == "fp4":
        return FP4Quantizer.dequantize(quantized_data)
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")


def apply_4bit_quantization(
    state_dict: Dict[str, torch.Tensor],
    config: QuantizationConfig
) -> Dict[str, Any]:
    """
    Apply 4-bit quantization to model state dict.
    
    Args:
        state_dict: Model state dictionary
        config: Quantization configuration
    
    Returns:
        Quantized state dictionary with metadata
    """
    quantized_dict = {}
    
    for key, tensor in state_dict.items():
        # Only quantize linear layer weights (not biases or norms)
        if 'weight' in key and 'norm' not in key and 'embedding' not in key:
            if tensor.dim() >= 2:  # Only quantize matrices
                quantized = quantize_linear_weight(tensor, config.bnb_4bit_quant_type)
                quantized_dict[key] = {
                    '_quantized': True,
                    'data': quantized
                }
            else:
                quantized_dict[key] = tensor
        else:
            quantized_dict[key] = tensor
    
    return quantized_dict


def dequantize_state_dict(
    quantized_dict: Dict[str, Any],
    quant_type: str = "nf4"
) -> Dict[str, torch.Tensor]:
    """
    Dequantize a quantized state dict.
    
    Args:
        quantized_dict: Quantized state dictionary
        quant_type: 'nf4' or 'fp4'
    
    Returns:
        Dequantized state dictionary
    """
    state_dict = {}
    
    for key, value in quantized_dict.items():
        if isinstance(value, dict) and value.get('_quantized'):
            state_dict[key] = dequantize_linear_weight(value['data'], quant_type)
        else:
            state_dict[key] = value
    
    return state_dict


class QuantizedLinear(nn.Module):
    """
    Linear layer with 4-bit quantized weights.
    
    Mimics bitsandbytes.nn.Linear4bit behavior.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        
        # Quantized weights stored as parameters
        self.register_buffer('quantized_weight', torch.zeros(
            (out_features, in_features // 2 + in_features % 2),
            dtype=torch.uint8
        ))
        self.register_buffer('weight_absmax', torch.zeros(out_features))
        self.register_buffer('weight_shape', torch.tensor([out_features, in_features]))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization on-the-fly."""
        # Dequantize weights
        weight_data = {
            'quantized': self.quantized_weight,
            'absmax': self.weight_absmax,
            'original_shape': tuple(self.weight_shape.tolist()),
            'original_dtype': self.compute_dtype,
            'block_size': 64,
            'numel': self.weight_shape[0].item() * self.weight_shape[1].item()
        }
        
        weight = dequantize_linear_weight(weight_data, self.quant_type).to(self.compute_dtype)
        
        # Linear operation
        return torch.nn.functional.linear(x.to(self.compute_dtype), weight, self.bias)
    
    def quantize_weight(self, weight: torch.Tensor):
        """Quantize and store a weight tensor."""
        quantized = quantize_linear_weight(weight, self.quant_type)
        
        self.quantized_weight = quantized['quantized']
        self.weight_absmax = quantized['absmax']
        self.weight_shape = torch.tensor(quantized['original_shape'])
