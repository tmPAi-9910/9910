"""
TmpAi Lite - Base Model Module

Defines the abstract base class and configuration for all Lite model implementations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    vocab_size: int = 32000
    embed_dim: int = 2048
    num_layers: int = 16
    num_heads: int = 16
    ff_dim: int = 8192
    max_seq_len: int = 4096
    dropout: float = 0.1
    pad_token_id: int = 0
    use_context_retention: bool = False


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all TmpAi Lite models.
    
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
    
    def save_pretrained(self, path: str):
        """Save model to a directory."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save state dict
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        
        # Save config
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'embed_dim': self.config.embed_dim,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'ff_dim': self.config.ff_dim,
            'max_seq_len': self.config.max_seq_len,
            'dropout': self.config.dropout,
            'pad_token_id': self.config.pad_token_id,
            'use_context_retention': self.config.use_context_retention,
        }
        import json
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_pretrained(cls, path: str, quantization_config=None, **kwargs):
        """Load model from a directory."""
        import os
        import json
        
        # Load config
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        else:
            config = ModelConfig()
        
        # Create model
        model = cls(config=config, **kwargs)
        
        # Load state dict
        model_path = os.path.join(path, 'pytorch_model.bin')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Apply quantization if configured
            if quantization_config is not None and quantization_config.load_in_4bit:
                from tmpai_lite.quantization import apply_4bit_quantization
                state_dict = apply_4bit_quantization(state_dict, quantization_config)
            
            model.load_state_dict(state_dict, strict=False)
        
        return model
    
    @classmethod
    def from_pretrained(cls, path: str, quantization_config=None, **kwargs):
        """Alias for load_pretrained."""
        return cls.load_pretrained(path, quantization_config=quantization_config, **kwargs)
