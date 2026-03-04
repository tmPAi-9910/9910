"""Interaction module."""

from tmpai.src.interaction.interface import (
    InteractionProtocol,
    Conversation,
    ConversationMessage,
    FeedbackCollector,
    ResponseQualityMonitor,
    StreamingInterface
)

__all__ = [
    'InteractionProtocol',
    'Conversation',
    'ConversationMessage',
    'FeedbackCollector',
    'ResponseQualityMonitor',
    'StreamingInterface'
]
