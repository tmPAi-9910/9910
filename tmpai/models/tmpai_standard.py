"""
TmpAi Standard 1.0 - Core Architecture Module

The original TmpAi model with transformer-based architecture and
enhanced context retention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from tmpai.models.base import BaseModel, ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional context retention improvements."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_cache: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_cache = use_cache
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Context retention enhancement: learned position bias
        self.context_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if cache is not None and self.use_cache:
            k = torch.cat([cache.get('k', torch.empty(0, self.num_heads, 0, self.head_dim, device=x.device)), k], dim=2)
            v = torch.cat([cache.get('v', torch.empty(0, self.num_heads, 0, self.head_dim, device=x.device)), v], dim=2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply context bias for retention enhancement
        attn_weights = attn_weights + self.context_bias
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)
        
        new_cache = None
        if self.use_cache:
            new_cache = {'k': k, 'v': v}
        
        return output, new_cache


class FeedForward(nn.Module):
    """Feed-forward network with activation and dropout."""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextRetentionLayer(nn.Module):
    """Specialized layer for improved context retention across long sequences."""
    
    def __init__(self, embed_dim: int, retention_window: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.retention_window = retention_window
        
        self.memory_bank = nn.Parameter(torch.randn(retention_window, embed_dim))
        self.memory_gate = nn.Linear(embed_dim, embed_dim)
        self.update_gate = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute attention over memory bank
        memory_attn = torch.matmul(x, self.memory_bank.T)
        memory_attn = F.softmax(memory_attn / math.sqrt(self.embed_dim), dim=-1)
        retrieved_memory = torch.matmul(memory_attn, self.memory_bank)
        
        # Gate for blending input with retrieved memory
        gate = torch.sigmoid(self.memory_gate(x))
        output = gate * x + (1 - gate) * retrieved_memory
        
        return output


class TransformerBlock(nn.Module):
    """Transformer decoder block with context retention enhancements."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_context_retention: bool = True
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        self.use_context_retention = use_context_retention
        if use_context_retention:
            self.context_retention = ContextRetentionLayer(embed_dim)
            self.norm_context = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Self-attention
        attn_out, new_cache = self.attention(x, mask, cache)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Context retention enhancement
        if self.use_context_retention:
            context_out = self.context_retention(x)
            x = self.norm_context(x + context_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, new_cache


class TmpAiModel(BaseModel):
    """
    TmpAi Standard 1.0 - Main Model Architecture
    
    Transformer-based LLM with enhanced context retention mechanisms,
    inspired by Claude Opus 4.6 architecture principles.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        ff_dim: int = 16384,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        use_context_retention: bool = True,
        pad_token_id: int = 0,
        config: Optional[ModelConfig] = None
    ):
        if config is None:
            config = ModelConfig(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
                pad_token_id=pad_token_id
            )
        
        super().__init__(config)
        
        self.use_context_retention = use_context_retention
        self.ff_dim = ff_dim
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.ff_dim,
                config.dropout,
                use_context_retention
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.norm_out = nn.LayerNorm(config.embed_dim)
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Tie output projection with token embedding for efficiency
        self.output_projection.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        x = self.dropout(token_embeds + pos_embeds)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attn_mask = self.create_attention_mask(seq_len, input_ids.device)
        else:
            attn_mask = attention_mask
        
        new_cache = {}
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            block_cache = cache.get(i) if cache is not None else None
            x, block_new_cache = block(x, attn_mask, block_cache)
            if use_cache:
                new_cache[i] = block_new_cache
        
        x = self.norm_out(x)
        logits = self.output_projection(x)
        
        return {
            'logits': logits,
            'cache': new_cache if use_cache else None
        }
    
    @torch.no_grad()
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
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, only sample from top k tokens
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids.clone()
        cache = None
        
        for _ in range(max_new_tokens):
            # Forward pass with cache
            outputs = self.forward(generated, use_cache=True, cache=cache)
            logits = outputs['logits']
            cache = outputs['cache']
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


def model_size(model: TmpAiModel) -> Dict[str, Any]:
    """Calculate model size statistics."""
    return model.get_model_info()
