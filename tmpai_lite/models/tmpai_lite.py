"""
TmpAi Lite - Core Architecture Module

Lightweight transformer-based LLM with ~1.6B parameters.
Simplified architecture without context retention for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from tmpai_lite.models.base import BaseModel, ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional KV cache support."""
    
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
    """Feed-forward network with GELU activation."""
    
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


class TransformerBlock(nn.Module):
    """Simplified transformer decoder block without context retention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Self-attention with residual
        attn_out, new_cache = self.attention(x, mask, cache)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, new_cache


class TmpAiLite(BaseModel):
    """
    TmpAi Lite - Lightweight LLM (~1.6B parameters)
    
    Simplified transformer architecture:
    - vocab_size: 32000
    - embed_dim: 2048
    - num_layers: 16
    - num_heads: 16
    - ff_dim: 8192
    - max_seq_len: 4096
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 2048,
        num_layers: int = 16,
        num_heads: int = 16,
        ff_dim: int = 8192,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        use_context_retention: bool = False,
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
                pad_token_id=pad_token_id,
                use_context_retention=use_context_retention
            )
        
        super().__init__(config)
        
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
                config.dropout
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
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
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
            eos_token_id: Token ID to stop generation
            pad_token_id: Token ID for padding
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
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
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
