# TmpAi Standard 1.0 Documentation

## Table of Contents

1. [Architecture](#architecture)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [User Interaction](#user-interaction)
5. [Safety](#safety)
6. [Deployment](#deployment)

---

## Architecture

### Model Architecture

TmpAi Standard 1.0 uses a transformer-based architecture with enhanced context retention mechanisms. The model implementations are organized in the `tmpai/models/` directory:

- **Embedding Dimension**: 4096
- **Layers**: 32 transformer blocks
- **Attention Heads**: 32 per layer
- **Feed-Forward Dimension**: 16384
- **Maximum Sequence Length**: 8192 tokens

### Model Structure

```
tmpai/models/
├── base.py            # Abstract base class (BaseModel) and ModelConfig
├── tmpai_standard.py  # TmpAi Standard 1.0 implementation
└── __init__.py        # Module exports
```

### Key Components

#### BaseModel

All models inherit from `BaseModel`, which defines:
- Common model configuration via `ModelConfig`
- Abstract methods: `forward()` and `generate()`
- Shared utility methods like `create_attention_mask()` and `get_model_info()`

#### Multi-Head Attention with Context Retention

The model uses an enhanced multi-head attention mechanism with:
- Learned position bias for improved context understanding
- Optional caching for efficient autoregressive generation
- Context retention layers for long-range dependencies

#### Context Retention Layer

A specialized layer that:
- Maintains a memory bank of past representations
- Uses gating mechanisms to blend input with retrieved context
- Improves performance on long-context tasks

### Usage

```python
from tmpai import TmpAiModel, BaseModel, ModelConfig
import torch

# Initialize model with explicit parameters
model = TmpAiModel(
    vocab_size=50000,
    embed_dim=4096,
    num_layers=32,
    num_heads=32,
    max_seq_len=8192
)

# Or using ModelConfig
config = ModelConfig(
    vocab_size=50000,
    embed_dim=4096,
    num_layers=32,
    num_heads=32,
    max_seq_len=8192
)
model = TmpAiModel(config=config)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Generate text
input_ids = torch.randint(0, 50000, (1, 100)).to(device)
output = model.generate(input_ids, max_new_tokens=100)
```

---

## Training

### Tiered Training Strategy

TmpAi uses a three-phase training approach:

1. **Pre-training**: General language comprehension on diverse datasets
2. **Fine-tuning**: Domain-specific adaptation
3. **RLHF**: Reinforcement learning from human feedback

### Training Configuration

```python
from tmpai.src.training import Trainer

# Create trainer
trainer = Trainer(model, config={
    'batch_size': 4,
    'learning_rate': 1e-4,
    'warmup_steps': 2000,
    'max_steps': 100000,
    'gradient_accumulation_steps': 4,
    'mixed_precision': True
})

# Pre-train
trainer.pretrain(train_dataset, eval_dataset)

# Fine-tune on specific domain
trainer.finetune(domain_dataset, domain='science')

# RLHF (with reward model)
trainer.rlhf(prompts, reward_model, num_steps=5000)
```

### Curriculum Learning

Training follows a curriculum that:
- Starts with general language modeling
- Progresses to domain-specific tasks
- Ends with reinforcement learning for alignment

---

## Evaluation

### Evaluation Metrics

TmpAi Standard 1.0 is evaluated on multiple dimensions:

- **Perplexity**: Language modeling performance
- **Accuracy**: Token-level prediction accuracy
- **BLEU**: Text generation quality
- **ROUGE**: Text similarity metrics
- **Context Retention**: Long-context understanding
- **User Satisfaction**: Simulated and real user feedback

### Running Evaluation

```python
from tmpai.src.evaluation import Evaluator

# Define benchmarks (e.g., Claude Opus 4.6 scores)
benchmarks = {
    'perplexity': 15.0,
    'bleu': 0.35,
    'rouge': 0.40
}

# Create evaluator
evaluator = Evaluator(model, benchmarks, device)

# Run full evaluation
results = evaluator.run_full_evaluation(test_data, save_path='results.json')

# Print report
evaluator.print_report(results)
```

### Benchmark Comparison

Results are compared against baseline models:
- Perplexity improvement percentage
- BLEU/ROUGE score gains
- Context retention metrics
- User satisfaction scores

---

## User Interaction

### Conversation Management

```python
from tmpai.src.interaction import InteractionProtocol

# Create interaction protocol
interaction = InteractionProtocol(model, max_context_length=4096)

# Create conversation with system prompt
conv_id = interaction.create_conversation(
    system_prompt="You are a helpful AI assistant."
)

# Send messages
response = interaction.send_message(
    conversation_id=conv_id,
    user_message="What is quantum physics?",
    generation_params={
        'temperature': 0.7,
        'max_new_tokens': 200
    }
)

print(f"Assistant: {response}")
```

### Feedback Collection

```python
# Submit feedback on a response
interaction.submit_feedback(
    conversation_id=conv_id,
    message_index=0,
    rating=5,
    comment="Very helpful explanation!"
)

# Get quality report
report = interaction.get_quality_report()
print(report)
```

### Streaming Interface

```python
from tmpai.src.interaction import StreamingInterface

streamer = StreamingInterface(model, interaction)

def on_chunk(chunk: str):
    print(chunk, end='', flush=True)

response = streamer.stream_response(
    conversation_id=conv_id,
    user_message="Tell me a story.",
    callback=on_chunk
)
```

---

## Safety

### Safety System

The safety system includes:

- **Content Filtering**: Detects harmful or inappropriate content
- **Bias Analysis**: Identifies and mitigates biased outputs
- **Personal Information Redaction**: Protects privacy
- **Refusal Mechanisms**: Appropriate responses to policy violations

### Using Safety System

```python
from tmpai.src.safety import SafetySystem

# Initialize safety system
safety = SafetySystem(model)

# Check input
input_check = safety.check_input(user_message)
if not input_check['is_safe']:
    refusal_msg = safety.generate_refusal_message(
        input_check['violations'][0].violation_type
    )
    print(refusal_msg)

# Check output
output_check = safety.check_output(generated_text)

# Redact sensitive content
redacted_text, redacted_items = safety.redact_content(text)

# Get safety report
report = safety.get_safety_report()
```

### Bias Mitigation

```python
from tmpai.src.safety import BiasAnalyzer

analyzer = BiasAnalyzer()

# Analyze bias in text
bias_analysis = analyzer.analyze_bias(text)

# Get improvement suggestions
suggestions = analyzer.suggest_improvements(text)
```

---

## Deployment

### Deployment Configuration

```python
from tmpai.src.deployment import DeploymentManager, DeploymentConfig

config = DeploymentConfig(
    model_path='model/',
    device='cuda',
    batch_size=1,
    api_port=8000,
    enable_streaming=True,
    auth_enabled=True
)

manager = DeploymentManager(model, config)
```

### Docker Deployment

```python
# Deploy with Docker
manager.deploy_docker(tag='tmpai-standard:1.0')

# This generates:
# - Dockerfile
# - docker-compose.yml
# - Builds the Docker image
```

Start services:
```bash
docker-compose up -d
```

### Kubernetes Deployment

```python
# Deploy to Kubernetes
manager.deploy_kubernetes()

# This generates:
# - deployment.yaml
# - service.yaml
# - hpa.yaml (HorizontalPodAutoscaler)
```

Apply manifests:
```bash
kubectl apply -f k8s/
```

### Model Export

```python
from tmpai.src.deployment import ModelExporter

exporter = ModelExporter(model)

# Export to PyTorch
exporter.export_pytorch('model/model_pytorch.pt')

# Export to ONNX (for cross-platform)
exporter.export_onnx('model/model.onnx')

# Export to TorchScript
exporter.export_torchscript('model/model_torchscript.pt')

# Quantize model (int8)
exporter.quantize_model('model/model_quantized.pt', dtype='int8')
```

### Continuous Learning

```python
# Setup continuous learning
manager.setup_continuous_learning(check_interval_hours=24)

# The system will:
# 1. Collect feedback from interactions
# 2. Analyze feedback patterns
# 3. Schedule updates when warranted
# 4. Create new model versions
```

### Monitoring

The deployment includes:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Health Checks**: Automatic health monitoring
- **Auto-scaling**: Kubernetes HPA for load scaling

Access monitoring:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## API Server (Coming Soon)

A REST API server will be provided for easy integration:

```python
# Start API server
python -m tmpai.deployment.api_server

# Example API calls
curl http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## Configuration Files

### Model Configuration

```yaml
# config/model.yaml
model:
  vocab_size: 50000
  embed_dim: 4096
  num_layers: 32
  num_heads: 32
  ff_dim: 16384
  max_seq_len: 8192
  dropout: 0.1
  use_context_retention: true
```

### Training Configuration

```yaml
# config/training.yaml
training:
  batch_size: 4
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 2000
  max_steps: 100000
  gradient_accumulation_steps: 4
  mixed_precision: true
  checkpoint_dir: "checkpoints"
```

---

## Best Practices

### Training

1. Start with pre-trained weights when possible
2. Use mixed precision training for efficiency
3. Monitor perplexity and validation loss
4. Implement early stopping based on validation metrics

### Inference

1. Use caching for autoregressive generation
2. Batch requests when possible
3. Implement streaming for better UX
4. Monitor generation quality

### Safety

1. Always run safety checks on inputs and outputs
2. Implement content filtering
3. Regularly review and update safety guidelines
4. Collect and act on user feedback

### Deployment

1. Use quantized models for edge deployment
2. Implement proper authentication and rate limiting
3. Set up monitoring and alerting
4. Use containerization for reproducibility

---

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Use mixed precision training
- Use model quantization

### Slow Inference

- Enable KV caching
- Use GPU acceleration
- Reduce sequence length
- Use TorchScript or ONNX export

### Quality Issues

- Increase model size
- Train on more data
- Adjust temperature and sampling parameters
- Fine-tune on domain-specific data

---

## Creating Custom Models

The framework supports adding new model implementations by inheriting from `BaseModel`:

### Step 1: Create a New Model File

Create a new file in `tmpai/models/` (e.g., `tmpai/models/custom_model.py`):

```python
"""
Custom Model Implementation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from tmpai.models.base import BaseModel, ModelConfig


class CustomModel(BaseModel):
    """
    A custom model implementation.
    
    Inherit from BaseModel and implement the required abstract methods.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        if config is None:
            config = ModelConfig(**kwargs)
        super().__init__(config)
        
        # Initialize your custom layers
        self.custom_layer = nn.Linear(config.embed_dim, config.embed_dim)
        
        self._init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementation.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            cache: Optional cached states
            use_cache: Whether to return cache
        
        Returns:
            Dictionary with 'logits' and optionally 'cache'
        """
        # Implement forward logic
        logits = self.custom_layer(input_ids.float())
        
        return {
            'logits': logits,
            'cache': None
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
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty
        
        Returns:
            Generated token IDs
        """
        self.eval()
        # Implement generation logic
        return input_ids
```

### Step 2: Export the Model

Add the import to `tmpai/models/__init__.py`:

```python
from tmpai.models.custom_model import CustomModel

__all__ = [
    'BaseModel',
    'ModelConfig',
    'TmpAiModel',
    'model_size',
    'CustomModel'  # Add your model here
]
```

### Step 3: Use the Model

```python
from tmpai import CustomModel, ModelConfig

# Create model
config = ModelConfig(vocab_size=50000, embed_dim=512)
model = CustomModel(config)

# Use with framework components
from tmpai.src.training import Trainer
trainer = Trainer(model)
```

---

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

---

## License

MIT License - see LICENSE file for details.
