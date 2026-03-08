# TmpAi

A large language model (LLM) framework with multiple model variants.

## Overview

This repository contains:
- **TmpAi Standard 1.0** (9910): Full-scale transformer model (~13B parameters)
- **TmpAi Lite** (0199): Lightweight model with built-in tokenizer (~1.6B parameters)

## TmpAi Standard 1.0

A comprehensive LLM framework designed to balance performance and resource efficiency through transformer-based architecture with innovative mechanisms for improved context retention.

### Features

- **Advanced Architecture**: Transformer-based neural network with attention mechanisms optimized for context retention
- **Multilingual Support**: Training on diverse datasets spanning multiple languages and domains
- **Tiered Training**: Transfer learning and reinforcement learning approach with domain-specific fine-tuning
- **Comprehensive Evaluation**: Performance metrics including perplexity, accuracy, and user satisfaction
- **User-Centric Design**: Intuitive interaction protocols with feedback collection
- **Ethical AI**: Built-in bias mitigation and content filtering systems

### Quick Start

```python
from tmpai import TmpAiModel
from tmpai.src.training import Trainer
from tmpai.src.evaluation import Evaluator

# Initialize model
model = TmpAiModel(
    vocab_size=50000,
    embed_dim=4096,
    num_layers=32,
    num_heads=32
)

# Train
trainer = Trainer(model)
trainer.pretrain(train_dataset, eval_dataset)

# Evaluate
evaluator = Evaluator(model, benchmarks, device)
results = evaluator.run_full_evaluation(test_data, save_path='results.json')
```

---

## TmpAi Lite

A lightweight 1.6B parameter language model with integrated tokenizer and 4-bit quantization support. **No external dependencies** - completely self-contained.

### Features

- **Lightweight**: ~1.6B parameters (1/8 of Standard)
- **Built-in Tokenizer**: BPE tokenizer with 32k vocabulary, no external tokenizer repo needed
- **4-bit Quantization**: NF4 quantization support for reduced memory usage
- **Japanese & English**: Multilingual support out of the box
- **Standalone**: Zero external dependencies for tokenizer

### Specifications

| Parameter | Standard (9910) | Lite (0199) |
|-----------|----------------|-------------|
| embed_dim | 4096 | 2048 |
| num_layers | 32 | 16 |
| num_heads | 32 | 16 |
| ff_dim | 16384 | 8192 |
| max_seq_len | 8192 | 4096 |
| vocab_size | 50000 | 32000 |
| Parameters | ~13B | ~1.6B |

### Quick Start

```python
from tmpai_lite import TmpAiLite, TmpAiLiteTokenizer, QuantizationConfig

# Initialize tokenizer
tokenizer = TmpAiLiteTokenizer()

# Initialize model
model = TmpAiLite()

# Encode text
tokens = tokenizer.encode("Hello, world!")

# Generate text
import torch
input_ids = torch.tensor([tokens])
output = model.generate(input_ids, max_new_tokens=50)
generated_text = tokenizer.decode(output[0].tolist())
```

### 4-bit Quantization

```python
from tmpai_lite import TmpAiLite, QuantizationConfig

# 4-bit quantization config
config = QuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
)

# Load model with quantization
model = TmpAiLite.from_pretrained(
    "path/to/model",
    quantization_config=config
)
```

### Save and Load

```python
# Save model and tokenizer
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_tokenizer")

# Load model and tokenizer
model = TmpAiLite.load_pretrained("./my_model")
tokenizer = TmpAiLiteTokenizer()
tokenizer.load_pretrained("./my_tokenizer")
```

## Project Structure

```
├── tmpai/                  # TmpAi Standard 1.0
│   ├── models/             # Model implementations
│   ├── src/                # Training, evaluation, etc.
│   └── ...
├── tmpai_lite/             # TmpAi Lite (0199)
│   ├── models/             # Lite model implementation
│   ├── tokenizer/          # Built-in BPE tokenizer
│   ├── quantization/       # 4-bit quantization
│   └── utils/              # Memory utilities
├── example.py              # Standard model example
├── example_lite.py         # Lite model example
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Examples

```bash
# TmpAi Standard 1.0
python example.py

# TmpAi Lite
python example_lite.py
```

## License

MIT License
