"""
Memory Utilities

Functions for monitoring and managing memory usage.
"""

import torch
from typing import Dict, Optional


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get memory usage statistics.
    
    Args:
        device: Device to check (defaults to CUDA if available, else CPU)
    
    Returns:
        Dictionary with memory statistics in bytes
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    stats = {
        'device': str(device),
        'allocated_bytes': 0,
        'reserved_bytes': 0,
        'free_bytes': 0,
        'total_bytes': 0,
    }
    
    if device.type == 'cuda':
        stats['allocated_bytes'] = torch.cuda.memory_allocated(device)
        stats['reserved_bytes'] = torch.cuda.memory_reserved(device)
        stats['total_bytes'] = torch.cuda.get_device_properties(device).total_memory
        stats['free_bytes'] = stats['total_bytes'] - stats['allocated_bytes']
    else:
        # CPU memory - use psutil if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats['total_bytes'] = mem.total
            stats['free_bytes'] = mem.available
            stats['allocated_bytes'] = mem.used
        except ImportError:
            stats['total_bytes'] = 0
            stats['free_bytes'] = 0
            stats['allocated_bytes'] = 0
    
    return stats


def format_memory_size(bytes_value: float) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        bytes_value: Number of bytes
    
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def print_memory_summary(device: Optional[torch.device] = None):
    """Print memory usage summary."""
    stats = get_memory_usage(device)
    
    print(f"Memory Usage ({stats['device']}):")
    print(f"  Allocated: {format_memory_size(stats['allocated_bytes'])}")
    print(f"  Reserved:  {format_memory_size(stats['reserved_bytes'])}")
    print(f"  Free:      {format_memory_size(stats['free_bytes'])}")
    print(f"  Total:     {format_memory_size(stats['total_bytes'])}")


def get_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    """
    Get memory usage for a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model memory statistics
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        'param_bytes': param_size,
        'buffer_bytes': buffer_size,
        'total_bytes': param_size + buffer_size,
        'param_size': format_memory_size(param_size),
        'total_size': format_memory_size(param_size + buffer_size),
    }


def clear_cache(device: Optional[torch.device] = None):
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
