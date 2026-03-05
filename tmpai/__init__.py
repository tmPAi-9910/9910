"""
TmpAi Standard 1.0

A large language model (LLM) framework inspired by Claude Opus 4.6,
focusing on enhanced comprehension, contextual awareness, and user engagement.
"""

__version__ = "1.0.0"
__author__ = "TmpAi-9910"

# Import from new models directory
from tmpai.models import TmpAiModel, model_size, BaseModel, ModelConfig

__all__ = ['TmpAiModel', 'model_size', 'BaseModel', 'ModelConfig']
