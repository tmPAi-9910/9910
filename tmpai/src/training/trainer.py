"""
TmpAi Standard 1.0 - Training Methodology Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import math
from datetime import datetime
from collections import deque

from tmpai.models import TmpAiModel


class TextDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(
        self,
        data_path: str,
        vocab_size: int,
        max_seq_len: int = 8192,
        stride: int = 512
    ):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.stride = stride
        
        # Load data
        self.data = self._load_data()
        self.vocab_size = vocab_size
        
    def _load_data(self) -> List[int]:
        """Load and tokenize data."""
        # This is a placeholder - in production, implement proper tokenization
        # using a tokenizer like BPE or WordPiece
        data = []
        for file_path in self.data_path.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Simple character-level tokenization (replace with proper tokenizer)
                tokens = [ord(c) % self.vocab_size for c in text]
                data.extend(tokens)
        return data
    
    def __len__(self) -> int:
        return max(0, (len(self.data) - self.max_seq_len) // self.stride + 1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_seq_len + 1
        
        if end_idx > len(self.data):
            # Pad if needed
            segment = self.data[start_idx:]
            segment += [0] * (self.max_seq_len + 1 - len(segment))
        else:
            segment = self.data[start_idx:end_idx]
        
        x = torch.tensor(segment[:-1], dtype=torch.long)
        y = torch.tensor(segment[1:], dtype=torch.long)
        return x, y


class CurriculumScheduler:
    """Curriculum learning scheduler for tiered training."""
    
    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        self.current_stage = 0
        self.stage_steps = [stage['steps'] for stage in stages]
        self.cumulative_steps = np.cumsum(self.stage_steps)
    
    def get_stage(self, global_step: int) -> int:
        """Get current curriculum stage based on global step."""
        for i, threshold in enumerate(self.cumulative_steps):
            if global_step < threshold:
                return i
        return len(self.stages) - 1
    
    def get_config(self, global_step: int) -> Dict[str, Any]:
        """Get training config for current stage."""
        stage_idx = self.get_stage(global_step)
        return self.stages[stage_idx]


class RewardModel(nn.Module):
    """Reward model for reinforcement learning from human feedback (RLHF)."""
    
    def __init__(self, embed_dim: int = 4096, num_layers: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings.mean(dim=1))


class PPOTrainer:
    """Proximal Policy Optimization trainer for RLHF."""
    
    def __init__(
        self,
        model: TmpAiModel,
        reward_model: RewardModel,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.model = model
        self.reward_model = reward_model
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Store reference model for KL penalty
        self.ref_model = None
    
    def set_reference_model(self, ref_model: TmpAiModel):
        """Set reference model for computing KL divergence."""
        self.ref_model = ref_model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_rewards(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor
    ) -> torch.Tensor:
        """Compute rewards using reward model."""
        with torch.no_grad():
            # Get embeddings from model
            outputs = self.model.forward(torch.cat([prompts, responses], dim=1))
            embeddings = self.model.token_embedding(torch.cat([prompts, responses], dim=1))
            rewards = self.reward_model(embeddings)
        return rewards.squeeze(-1)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            batch: Dictionary containing prompts, responses, old_log_probs, advantages
            optimizer: Optimizer for policy updates
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        prompts = batch['prompts']
        responses = batch['responses']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        
        # Forward pass through model
        outputs = self.model.forward(torch.cat([prompts, responses], dim=1))
        logits = outputs['logits']
        
        # Compute new log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        response_log_probs = self._gather_response_log_probs(log_probs, prompts.size(1), responses)
        
        # Compute ratio
        ratio = torch.exp(response_log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL penalty (if reference model exists)
        kl_penalty = 0
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model.forward(torch.cat([prompts, responses], dim=1))
                ref_log_probs = F.log_softmax(ref_outputs['logits'], dim=-1)
            kl_div = (log_probs - ref_log_probs).mean(dim=-1)
            kl_penalty = kl_div.mean()
            policy_loss += 0.1 * kl_penalty
        
        # Entropy bonus
        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss - self.entropy_coef * entropy
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'kl_penalty': kl_penalty if isinstance(kl_penalty, float) else kl_penalty.item()
        }
    
    def _gather_response_log_probs(
        self,
        log_probs: torch.Tensor,
        prompt_len: int,
        responses: torch.Tensor
    ) -> torch.Tensor:
        """Gather log probabilities for response tokens."""
        batch_size, seq_len, vocab_size = log_probs.shape
        response_len = responses.size(1)
        
        # Get log probs for response tokens only
        response_log_probs = log_probs[:, prompt_len:prompt_len+response_len, :]
        
        # Gather the log probs for actual response tokens
        gathered = torch.gather(response_log_probs, 2, responses.unsqueeze(-1))
        return gathered.squeeze(-1).sum(dim=1)


class Trainer:
    """
    Main trainer for TmpAi Standard 1.0
    
    Implements tiered training strategy:
    1. General language comprehension (unsupervised pre-training)
    2. Domain-specific fine-tuning
    3. Reinforcement learning from human feedback
    """
    
    def __init__(
        self,
        model: TmpAiModel,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.config = config or self._default_config()
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.curriculum = self._setup_curriculum()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = {
            'loss': [],
            'perplexity': [],
            'learning_rate': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration."""
        return {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 2000,
            'max_steps': 100000,
            'gradient_accumulation_steps': 4,
            'max_grad_norm': 1.0,
            'eval_steps': 1000,
            'save_steps': 5000,
            'checkpoint_dir': 'checkpoints',
            'mixed_precision': True
        }
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with parameter groups."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'LayerNorm.weight' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.config['weight_decay']
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            }
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999)
        )
    
    def _setup_scheduler(self) -> Any:
        """Setup learning rate scheduler."""
        warmup_steps = self.config['warmup_steps']
        total_steps = self.config['max_steps']
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_curriculum(self) -> CurriculumScheduler:
        """Setup curriculum learning stages."""
        stages = [
            {
                'name': 'pre_training',
                'steps': 70000,
                'learning_rate': 1e-4,
                'data_path': 'data/general'
            },
            {
                'name': 'domain_finetuning',
                'steps': 25000,
                'learning_rate': 5e-5,
                'data_path': 'data/domains'
            },
            {
                'name': 'rlhf',
                'steps': 5000,
                'learning_rate': 1e-5,
                'data_path': 'data/feedback'
            }
        ]
        return CurriculumScheduler(stages)
    
    def pretrain(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from: Optional[str] = None
    ) -> None:
        """
        Unsupervised pre-training phase.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from: Optional checkpoint path to resume from
        """
        if resume_from:
            self._load_checkpoint(resume_from)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.model.train()
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config['mixed_precision'] else None
        
        accumulated_loss = 0
        accumulation_steps = self.config['gradient_accumulation_steps']
        
        for epoch in range(self.config.get('num_epochs', 1)):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if self.global_step >= self.config['max_steps']:
                    break
                
                inputs = inputs.to(self.model.token_embedding.weight.device)
                targets = targets.to(self.model.token_embedding.weight.device)
                
                # Forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.forward(inputs)
                        logits = outputs['logits']
                        
                        # Compute loss
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = targets[..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=self.model.pad_token_id
                        )
                        loss = loss / accumulation_steps
                else:
                    outputs = self.model.forward(inputs)
                    logits = outputs['logits']
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=self.model.pad_token_id
                    )
                    loss = loss / accumulation_steps
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                
                # Update weights
                if (batch_idx + 1) % accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Log metrics
                    perplexity = math.exp(accumulated_loss * accumulation_steps)
                    self.training_history['loss'].append(accumulated_loss * accumulation_steps)
                    self.training_history['perplexity'].append(perplexity)
                    self.training_history['learning_rate'].append(self.scheduler.get_last_lr()[0])
                    
                    print(f"Step {self.global_step}: Loss={accumulated_loss * accumulation_steps:.4f}, "
                          f"PPL={perplexity:.2f}, LR={self.scheduler.get_last_lr()[0]:.2e}")
                    
                    accumulated_loss = 0
                    self.global_step += 1
                    
                    # Evaluation
                    if eval_dataset and self.global_step % self.config['eval_steps'] == 0:
                        eval_results = self.evaluate(eval_dataset)
                        print(f"Evaluation: {eval_results}")
                        
                        # Save best model
                        if eval_results['loss'] < self.best_loss:
                            self.best_loss = eval_results['loss']
                            self._save_checkpoint('best_model')
                    
                    # Save checkpoint
                    if self.global_step % self.config['save_steps'] == 0:
                        self._save_checkpoint(f'checkpoint_{self.global_step}')
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model on evaluation dataset."""
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs = inputs.to(self.model.token_embedding.weight.device)
                targets = targets.to(self.model.token_embedding.weight.device)
                
                outputs = self.model.forward(inputs)
                logits = outputs['logits']
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.model.pad_token_id,
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += (shift_labels != self.model.pad_token_id).sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def finetune(
        self,
        train_dataset: Dataset,
        domain: str = 'science',
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Domain-specific fine-tuning phase.
        
        Args:
            train_dataset: Training dataset for specific domain
            domain: Domain name for logging
            learning_rate: Optional custom learning rate
        """
        if learning_rate:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        print(f"Fine-tuning on {domain} domain with LR={learning_rate or self.config['learning_rate']}")
        self.pretrain(train_dataset)
    
    def rlhf(
        self,
        prompts: List[str],
        reward_model: RewardModel,
        num_steps: int = 5000
    ) -> None:
        """
        Reinforcement Learning from Human Feedback phase.
        
        Args:
            prompts: List of prompt strings
            reward_model: Trained reward model
            num_steps: Number of PPO steps
        """
        ppo_trainer = PPOTrainer(self.model, reward_model)
        ppo_trainer.set_reference_model(self.model)
        
        print(f"Starting RLHF for {num_steps} steps")
        
        for step in range(num_steps):
            # Sample batch of prompts
            batch_prompts = np.random.choice(prompts, size=self.config['batch_size'])
            
            # Generate responses
            prompt_tokens = [self._tokenize(p) for p in batch_prompts]
            # ... implement generation logic
            
            # Compute rewards
            # ... implement reward computation
            
            # PPO update
            # ... implement PPO step
            
            if step % 100 == 0:
                print(f"RLHF step {step}/{num_steps}")
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text (placeholder - implement proper tokenization)."""
        tokens = [ord(c) % self.model.vocab_size for c in text]
        return torch.tensor(tokens, dtype=torch.long)
    
    def _save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        print(f"Loaded checkpoint from {path}")
