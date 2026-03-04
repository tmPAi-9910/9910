"""
TmpAi Models - Base Model Module

Defines the abstract base class and configuration for all model implementations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    vocab_size: int = 50000
    embed_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    ff_dim: int = 16384
    max_seq_len: int = 8192
    dropout: float = 0.1
    pad_token_id: int = 0


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all TmpAi models.
    
    All model implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.num_layers = config.num_layers
        self.max_seq_len = config.max_seq_len
        self.pad_token_id = config.pad_token_id
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            cache: Optional cached key-value states for generation
            use_cache: Whether to return cache for autoregressive generation
        
        Returns:
            Dictionary containing logits and optional cache
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        pass
    
    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        return torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    def _init_weights(self):
        """Initialize model weights following best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        param_count = sum(p.numel() for p in self.parameters())
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.__class__.__name__,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'total_parameters': param_count,
            'trainable_parameters': trainable_count,
            'parameter_count_str': f'{param_count:,}',
            'model_size_gb': param_count * 4 / (1024**3)  # Assuming float32
        }
