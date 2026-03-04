"""
TmpAi Standard 1.0 - User Interaction Module
"""

import torch
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from collections import defaultdict, deque

from tmpai.models import TmpAiModel


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Represents a conversation between user and assistant."""
    conversation_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to the conversation."""
        self.messages.append(ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
    
    def get_context(self, max_messages: Optional[int] = None) -> str:
        """Get formatted conversation context."""
        if max_messages:
            messages = self.messages[-max_messages:]
        else:
            messages = self.messages
        
        context_parts = []
        for msg in messages:
            context_parts.append(f"{msg.role.capitalize()}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            'conversation_id': self.conversation_id,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
                    'metadata': msg.metadata
                }
                for msg in self.messages
            ],
            'metadata': self.metadata
        }


class FeedbackCollector:
    """Collects and manages user feedback for model improvement."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path('feedback')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.feedback_buffer: List[Dict[str, Any]] = []
    
    def collect_feedback(
        self,
        conversation_id: str,
        message_id: str,
        feedback_type: str,  # 'rating', 'correction', 'flag'
        feedback_value: Union[int, str, bool],
        comment: Optional[str] = None
    ) -> None:
        """
        Collect user feedback on a model response.
        
        Args:
            conversation_id: ID of the conversation
            message_id: ID of the specific message
            feedback_type: Type of feedback (rating, correction, flag)
            feedback_value: Value of the feedback (rating score, corrected text, etc.)
            comment: Optional additional comments
        """
        feedback_entry = {
            'conversation_id': conversation_id,
            'message_id': message_id,
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_buffer.append(feedback_entry)
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback."""
        if not self.feedback_buffer:
            return {}
        
        summary = {
            'total_feedback': len(self.feedback_buffer),
            'by_type': defaultdict(int),
            'average_rating': 0.0,
            'flag_count': 0
        }
        
        ratings = []
        for feedback in self.feedback_buffer:
            summary['by_type'][feedback['feedback_type']] += 1
            
            if feedback['feedback_type'] == 'rating':
                ratings.append(feedback['feedback_value'])
            elif feedback['feedback_type'] == 'flag':
                summary['flag_count'] += 1
        
        if ratings:
            summary['average_rating'] = sum(ratings) / len(ratings)
        
        return summary
    
    def save_feedback(self, filename: Optional[str] = None) -> None:
        """Save feedback to disk."""
        if not filename:
            filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.storage_path / filename
        with open(filepath, 'w') as f:
            json.dump(self.feedback_buffer, f, indent=2)
        
        self.feedback_buffer.clear()
        print(f"Feedback saved to {filepath}")
    
    def load_feedback(self, filepath: str) -> List[Dict[str, Any]]:
        """Load feedback from disk."""
        with open(filepath, 'r') as f:
            self.feedback_buffer = json.load(f)
        return self.feedback_buffer


class ResponseQualityMonitor:
    """Monitors response quality in real-time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_buffer = deque(maxlen=window_size)
    
    def record_response(
        self,
        prompt: str,
        response: str,
        generation_time: float,
        token_count: int
    ) -> None:
        """Record response metrics."""
        self.metrics_buffer.append({
            'prompt_length': len(prompt.split()),
            'response_length': len(response.split()),
            'generation_time': generation_time,
            'tokens_per_second': token_count / generation_time if generation_time > 0 else 0,
            'token_count': token_count
        })
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get aggregated quality metrics."""
        if not self.metrics_buffer:
            return {}
        
        metrics = list(self.metrics_buffer)
        
        return {
            'avg_response_length': sum(m['response_length'] for m in metrics) / len(metrics),
            'avg_generation_time': sum(m['generation_time'] for m in metrics) / len(metrics),
            'avg_tokens_per_second': sum(m['tokens_per_second'] for m in metrics) / len(metrics),
            'total_responses': len(metrics)
        }


class InteractionProtocol:
    """
    Defines user interaction protocols for TmpAi Standard 1.0.
    
    Emphasizes intuitive communication, responsiveness, and
    continuous feedback collection.
    """
    
    def __init__(
        self,
        model: TmpAiModel,
        max_context_length: int = 4096,
        feedback_collector: Optional[FeedbackCollector] = None
    ):
        self.model = model
        self.max_context_length = max_context_length
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.quality_monitor = ResponseQualityMonitor()
        self.conversations: Dict[str, Conversation] = {}
        self.conversation_counter = 0
        
    def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation.
        
        Args:
            system_prompt: Optional system prompt to set conversation context
            metadata: Optional metadata for the conversation
        
        Returns:
            Conversation ID
        """
        self.conversation_counter += 1
        conversation_id = f"conv_{self.conversation_counter}"
        
        conversation = Conversation(
            conversation_id=conversation_id,
            metadata=metadata or {}
        )
        
        if system_prompt:
            conversation.add_message('system', system_prompt)
        
        self.conversations[conversation_id] = conversation
        return conversation_id
    
    def send_message(
        self,
        conversation_id: str,
        user_message: str,
        generation_params: Optional[Dict[str, Any]] = None,
        collect_feedback: bool = True
    ) -> str:
        """
        Send a user message and get model response.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's message
            generation_params: Optional generation parameters
            collect_feedback: Whether to enable feedback collection for this message
        
        Returns:
            Model response
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        # Add user message to conversation
        conversation.add_message('user', user_message)
        
        # Prepare input
        context = self._prepare_input(conversation)
        
        # Generate response
        import time
        start_time = time.time()
        
        input_ids = self._tokenize(context).unsqueeze(0)
        input_ids = input_ids.to(self.model.token_embedding.weight.device)
        
        generation_params = generation_params or {}
        default_params = {
            'max_new_tokens': 256,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9
        }
        default_params.update(generation_params)
        
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids, **default_params)
        
        generation_time = time.time() - start_time
        
        # Extract response
        response = self._extract_response(generated_ids[0], context)
        
        # Add assistant response to conversation
        message_metadata = {}
        if collect_feedback:
            message_metadata['feedback_enabled'] = True
        conversation.add_message('assistant', response, message_metadata)
        
        # Record quality metrics
        token_count = len(response.split())
        self.quality_monitor.record_response(
            user_message, response, generation_time, token_count
        )
        
        return response
    
    def _prepare_input(self, conversation: Conversation) -> str:
        """Prepare input string from conversation context."""
        # Get recent messages within context limit
        messages = []
        total_length = 0
        
        for msg in reversed(conversation.messages):
            msg_length = len(msg.content.split())
            if total_length + msg_length > self.max_context_length:
                break
            messages.insert(0, msg)
            total_length += msg_length
        
        # Format messages
        formatted = []
        for msg in messages:
            if msg.role == 'system':
                formatted.append(f"System: {msg.content}")
            elif msg.role == 'user':
                formatted.append(f"User: {msg.content}")
            elif msg.role == 'assistant':
                formatted.append(f"Assistant: {msg.content}")
        
        context = "\n".join(formatted) + "\nAssistant:"
        return context
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text (placeholder - implement proper tokenization)."""
        tokens = [ord(c) % self.model.vocab_size for c in text]
        return torch.tensor(tokens, dtype=torch.long)
    
    def _extract_response(self, generated_ids: torch.Tensor, context: str) -> str:
        """Extract response from generated tokens."""
        # Simple detokenization (placeholder)
        tokens = [chr((tid % 128) + 32) for tid in generated_ids if tid > 0]
        response_text = ''.join(tokens)
        
        # Try to extract just the assistant's response
        if "Assistant:" in response_text:
            parts = response_text.split("Assistant:")
            # Get the last Assistant: section
            response = parts[-1].strip()
        else:
            response = response_text
        
        return response
    
    def submit_feedback(
        self,
        conversation_id: str,
        message_index: int,
        rating: Optional[int] = None,
        correction: Optional[str] = None,
        flag: bool = False,
        comment: Optional[str] = None
    ) -> None:
        """
        Submit feedback on a specific message.
        
        Args:
            conversation_id: ID of the conversation
            message_index: Index of the message in the conversation
            rating: Rating from 1-5 (optional)
            correction: Corrected text (optional)
            flag: Whether to flag the message as problematic
            comment: Additional comments (optional)
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        if message_index >= len(conversation.messages):
            raise ValueError(f"Message index {message_index} out of range")
        
        message = conversation.messages[message_index]
        message_id = f"{conversation_id}_{message_index}"
        
        if rating is not None:
            self.feedback_collector.collect_feedback(
                conversation_id, message_id, 'rating', rating, comment
            )
        
        if correction is not None:
            self.feedback_collector.collect_feedback(
                conversation_id, message_id, 'correction', correction, comment
            )
        
        if flag:
            self.feedback_collector.collect_feedback(
                conversation_id, message_id, 'flag', True, comment
            )
    
    def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a conversation by ID."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        return self.conversations[conversation_id]
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self.conversations.keys())
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get quality metrics report."""
        return {
            'response_quality': self.quality_monitor.get_quality_metrics(),
            'feedback_summary': self.feedback_collector.get_feedback_summary()
        }
    
    def save_conversation(self, conversation_id: str, filepath: str) -> None:
        """Save conversation to file."""
        conversation = self.get_conversation(conversation_id)
        with open(filepath, 'w') as f:
            json.dump(conversation.to_dict(), f, indent=2)
    
    def load_conversation(self, filepath: str) -> str:
        """Load conversation from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        conversation = Conversation(
            conversation_id=data['conversation_id'],
            metadata=data.get('metadata', {})
        )
        
        for msg_data in data['messages']:
            conversation.messages.append(ConversationMessage(
                role=msg_data['role'],
                content=msg_data['content'],
                timestamp=msg_data.get('timestamp'),
                metadata=msg_data.get('metadata', {})
            ))
        
        self.conversations[conversation.conversation_id] = conversation
        return conversation.conversation_id


class StreamingInterface:
    """Interface for streaming model responses."""
    
    def __init__(
        self,
        model: TmpAiModel,
        interaction_protocol: InteractionProtocol
    ):
        self.model = model
        self.protocol = interaction_protocol
    
    def stream_response(
        self,
        conversation_id: str,
        user_message: str,
        callback: Callable[[str], None],
        chunk_size: int = 10
    ) -> str:
        """
        Stream response generation with callback.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's message
            callback: Function to call with each chunk
            chunk_size: Number of tokens per chunk
        
        Returns:
            Complete response
        """
        # Get full response first (placeholder for true streaming)
        response = self.protocol.send_message(conversation_id, user_message)
        
        # Simulate streaming by calling callback with chunks
        words = response.split()
        current_chunk = []
        
        for i, word in enumerate(words):
            current_chunk.append(word)
            if len(current_chunk) >= chunk_size or i == len(words) - 1:
                callback(' '.join(current_chunk) + (' ' if i < len(words) - 1 else ''))
                current_chunk.clear()
        
        return response
