"""
TmpAi Standard 1.0 - Evaluation Metrics Module
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime

from tmpai.models import TmpAiModel


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for TmpAi Standard 1.0
    
    Metrics include:
    - Perplexity: Language modeling performance
    - Accuracy: Token-level prediction accuracy
    - BLEU: Text generation quality
    - ROUGE: Text similarity
    - Context Retention: Ability to maintain context over long sequences
    - User Satisfaction: Simulated user feedback
    """
    
    def __init__(self, model: TmpAiModel, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def compute_perplexity(self, dataset) -> float:
        """
        Compute perplexity on a dataset.
        
        Lower perplexity indicates better language modeling.
        """
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for inputs, targets in dataset:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
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
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(avg_loss)
        return perplexity
    
    def compute_accuracy(self, dataset) -> float:
        """
        Compute token-level prediction accuracy.
        """
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataset:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model.forward(inputs)
                logits = outputs['logits']
                
                predictions = torch.argmax(logits, dim=-1)
                
                mask = (targets != self.model.pad_token_id)
                correct += ((predictions == targets) & mask).sum().item()
                total += mask.sum().item()
        
        accuracy = correct / max(total, 1)
        return accuracy
    
    def compute_bleu_score(
        self,
        predictions: List[str],
        references: List[str],
        n_gram: int = 4
    ) -> Dict[str, float]:
        """
        Compute BLEU score for text generation quality.
        
        Returns BLEU scores for different n-gram orders.
        """
        from collections import Counter
        
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            return Counter([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
        
        def clip_counts(pred_counts: Counter, ref_counts: Counter) -> Counter:
            clipped = Counter()
            for ngram, count in pred_counts.items():
                clipped[ngram] = min(count, ref_counts.get(ngram, 0))
            return clipped
        
        scores = {}
        
        for n in range(1, n_gram + 1):
            pred_counts = Counter()
            ref_counts = Counter()
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                pred_counts.update(get_ngrams(pred_tokens, n))
                ref_counts.update(get_ngrams(ref_tokens, n))
            
            clipped = clip_counts(pred_counts, ref_counts)
            
            if sum(pred_counts.values()) == 0:
                scores[f'bleu_{n}'] = 0.0
            else:
                scores[f'bleu_{n}'] = sum(clipped.values()) / sum(pred_counts.values())
        
        # Geometric mean of n-gram precisions
        if all(scores[f'bleu_{n}'] > 0 for n in range(1, n_gram + 1)):
            scores['bleu'] = np.exp(np.mean([np.log(scores[f'bleu_{n}']) for n in range(1, n_gram + 1)]))
        else:
            scores['bleu'] = 0.0
        
        # Brevity penalty
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        if sum(pred_lengths) > sum(ref_lengths):
            bp = 1.0
        else:
            bp = np.exp(1 - sum(ref_lengths) / max(sum(pred_lengths), 1))
        
        scores['bleu_with_bp'] = scores['bleu'] * bp
        
        return scores
    
    def compute_rouge_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for text similarity.
        
        Returns ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        def get_ngrams(text: str, n: int) -> Counter:
            tokens = text.split()
            return Counter([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
        
        def lcs_length(a: List[str], b: List[str]) -> int:
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        scores = {
            'rouge_1_precision': 0.0,
            'rouge_1_recall': 0.0,
            'rouge_1_f1': 0.0,
            'rouge_2_precision': 0.0,
            'rouge_2_recall': 0.0,
            'rouge_2_f1': 0.0,
            'rouge_l_precision': 0.0,
            'rouge_l_recall': 0.0,
            'rouge_l_f1': 0.0
        }
        
        for pred, ref in zip(predictions, references):
            # ROUGE-1 (unigrams)
            pred_1grams = get_ngrams(pred, 1)
            ref_1grams = get_ngrams(ref, 1)
            
            overlap_1 = sum((pred_1grams & ref_1grams).values())
            pred_len_1 = len(pred.split())
            ref_len_1 = len(ref.split())
            
            scores['rouge_1_precision'] += overlap_1 / max(pred_len_1, 1)
            scores['rouge_1_recall'] += overlap_1 / max(ref_len_1, 1)
            
            # ROUGE-2 (bigrams)
            pred_2grams = get_ngrams(pred, 2)
            ref_2grams = get_ngrams(ref, 2)
            
            overlap_2 = sum((pred_2grams & ref_2grams).values())
            pred_len_2 = max(len(pred.split()) - 1, 1)
            ref_len_2 = max(len(ref.split()) - 1, 1)
            
            scores['rouge_2_precision'] += overlap_2 / pred_len_2
            scores['rouge_2_recall'] += overlap_2 / ref_len_2
            
            # ROUGE-L (longest common subsequence)
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            lcs = lcs_length(pred_tokens, ref_tokens)
            scores['rouge_l_precision'] += lcs / max(len(pred_tokens), 1)
            scores['rouge_l_recall'] += lcs / max(len(ref_tokens), 1)
        
        # Average over all predictions
        n = len(predictions)
        for key in scores:
            scores[key] /= n
        
        # Compute F1 scores
        for prefix in ['rouge_1', 'rouge_2', 'rouge_l']:
            p = scores[f'{prefix}_precision']
            r = scores[f'{prefix}_recall']
            if p + r > 0:
                scores[f'{prefix}_f1'] = 2 * p * r / (p + r)
            else:
                scores[f'{prefix}_f1'] = 0.0
        
        return scores
    
    def evaluate_context_retention(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate the model's ability to retain and use context.
        
        Args:
            contexts: Context paragraphs
            questions: Questions about the context
            answers: Ground truth answers
        
        Returns:
            Dictionary of context retention metrics
        """
        correct = 0
        total = len(questions)
        relevance_scores = []
        
        with torch.no_grad():
            for context, question, answer in zip(contexts, questions, answers):
                # Combine context and question
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                input_ids = self._tokenize(prompt).unsqueeze(0).to(self.device)
                
                # Generate response
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=False
                )
                
                generated_text = self._detokenize(generated[0])
                
                # Simple exact match for correctness
                if answer.lower() in generated_text.lower():
                    correct += 1
                
                # Relevance score (presence of key terms)
                key_terms = answer.lower().split()
                generated_lower = generated_text.lower()
                relevant_terms = sum(1 for term in key_terms if term in generated_lower)
                relevance = relevant_terms / len(key_terms) if key_terms else 0
                relevance_scores.append(relevance)
        
        return {
            'context_accuracy': correct / total,
            'context_relevance': np.mean(relevance_scores)
        }
    
    def evaluate_multilingual(
        self,
        test_data: Dict[str, List[Tuple[str, str]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multilingual capabilities.
        
        Args:
            test_data: Dictionary mapping language codes to list of (prompt, reference) tuples
        
        Returns:
            Dictionary mapping languages to their metrics
        """
        results = {}
        
        for lang, examples in test_data.items():
            predictions = []
            references = []
            
            for prompt, reference in examples:
                input_ids = self._tokenize(prompt).unsqueeze(0).to(self.device)
                generated = self.model.generate(input_ids, max_new_tokens=100)
                generated_text = self._detokenize(generated[0])
                
                predictions.append(generated_text)
                references.append(reference)
            
            bleu = self.compute_bleu_score(predictions, references)
            rouge = self.compute_rouge_score(predictions, references)
            
            results[lang] = {
                'bleu': bleu['bleu_with_bp'],
                'rouge_l': rouge['rouge_l_f1']
            }
        
        return results
    
    def simulate_user_satisfaction(
        self,
        prompts: List[str],
        num_samples: int = 5
    ) -> Dict[str, float]:
        """
        Simulate user satisfaction based on response quality.
        
        Evaluates responses on:
        - Relevance (response matches prompt intent)
        - Coherence (logical flow and consistency)
        - Helpfulness (useful and informative)
        - Safety (no harmful content)
        """
        satisfaction_scores = {
            'relevance': [],
            'coherence': [],
            'helpfulness': [],
            'safety': []
        }
        
        for prompt in prompts:
            for _ in range(num_samples):
                input_ids = self._tokenize(prompt).unsqueeze(0).to(self.device)
                generated = self.model.generate(input_ids, max_new_tokens=200)
                response = self._detokenize(generated[0])
                
                # Simulate scoring (in production, use real human feedback)
                satisfaction_scores['relevance'].append(self._score_relevance(prompt, response))
                satisfaction_scores['coherence'].append(self._score_coherence(response))
                satisfaction_scores['helpfulness'].append(self._score_helpfulness(prompt, response))
                satisfaction_scores['safety'].append(self._score_safety(response))
        
        return {
            key: np.mean(values) for key, values in satisfaction_scores.items()
        }
    
    def _score_relevance(self, prompt: str, response: str) -> float:
        """Score response relevance (0-1)."""
        # Check if response contains keywords from prompt
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words)
        return min(overlap / max(len(prompt_words), 1), 1.0)
    
    def _score_coherence(self, response: str) -> float:
        """Score response coherence (0-1)."""
        # Simple heuristic: check for logical connectors and sentence structure
        connectors = ['because', 'therefore', 'however', 'thus', 'consequently', 'moreover']
        response_lower = response.lower()
        connector_count = sum(1 for c in connectors if c in response_lower)
        
        # Length penalty (too short or too long may be less coherent)
        length_score = 1.0 - abs(len(response.split()) - 50) / 100
        length_score = max(0.5, min(1.0, length_score))
        
        return length_score * min(connector_count / 3 + 0.5, 1.0)
    
    def _score_helpfulness(self, prompt: str, response: str) -> float:
        """Score response helpfulness (0-1)."""
        # Check for informative content
        if len(response.split()) < 10:
            return 0.3
        
        # Check if response provides information beyond restating prompt
        prompt_set = set(prompt.lower().split())
        response_set = set(response.lower().split())
        new_info = len(response_set - prompt_set)
        
        return min(new_info / max(len(prompt_set), 1) + 0.5, 1.0)
    
    def _score_safety(self, response: str) -> float:
        """Score response safety (0-1)."""
        # Simple keyword-based safety check (in production, use comprehensive safety model)
        unsafe_keywords = [
            'violence', 'harm', 'illegal', 'hack', 'exploit',
            'hate', 'discriminate', 'weapon', 'drug'
        ]
        
        response_lower = response.lower()
        for keyword in unsafe_keywords:
            if keyword in response_lower:
                return 0.0
        
        return 1.0
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text (placeholder - implement proper tokenization)."""
        tokens = [ord(c) % self.model.vocab_size for c in text]
        return torch.tensor(tokens, dtype=torch.long)
    
    def _detokenize(self, token_ids: torch.Tensor) -> str:
        """Detokenize token IDs (placeholder - implement proper detokenization)."""
        tokens = [chr((tid % 128) + 32) for tid in token_ids if tid > 0]
        return ''.join(tokens)


class Evaluator:
    """
    Main evaluator for TmpAi Standard 1.0
    
    Orchestrates comprehensive evaluation with benchmarking against
    reference models (e.g., Claude Opus 4.6).
    """
    
    def __init__(
        self,
        model: TmpAiModel,
        benchmarks: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.metrics = EvaluationMetrics(model, device)
        self.benchmarks = benchmarks
        self.device = device or next(model.parameters()).device
    
    def run_full_evaluation(
        self,
        test_data: Any,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation suite.
        
        Returns comprehensive evaluation report.
        """
        print("Running full evaluation...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'embed_dim': self.model.embed_dim,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'max_seq_len': self.model.max_seq_len,
                'vocab_size': self.model.vocab_size
            }
        }
        
        # Core language metrics
        print("Computing perplexity...")
        results['perplexity'] = self.metrics.compute_perplexity(test_data)
        
        print("Computing accuracy...")
        results['accuracy'] = self.metrics.compute_accuracy(test_data)
        
        # Generation quality
        print("Evaluating generation quality...")
        predictions, references = self._generate_predictions(test_data)
        results['bleu'] = self.metrics.compute_bleu_score(predictions, references)
        results['rouge'] = self.metrics.compute_rouge_score(predictions, references)
        
        # Context retention
        print("Evaluating context retention...")
        contexts, questions, answers = self._load_context_test_data()
        results['context_retention'] = self.metrics.evaluate_context_retention(
            contexts, questions, answers
        )
        
        # User satisfaction
        print("Simulating user satisfaction...")
        prompts = self._load_satisfaction_prompts()
        results['user_satisfaction'] = self.metrics.simulate_user_satisfaction(prompts)
        
        # Compare with benchmarks
        results['benchmark_comparison'] = self._compare_with_benchmarks(results)
        
        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {save_path}")
        
        return results
    
    def _generate_predictions(
        self,
        test_data: Any,
        num_samples: int = 100
    ) -> Tuple[List[str], List[str]]:
        """Generate predictions for evaluation."""
        predictions = []
        references = []
        
        for i, (prompt, reference) in enumerate(test_data):
            if i >= num_samples:
                break
            
            input_ids = self.metrics._tokenize(prompt).unsqueeze(0).to(self.device)
            generated = self.model.generate(input_ids, max_new_tokens=100)
            generated_text = self.metrics._detokenize(generated[0])
            
            predictions.append(generated_text)
            references.append(reference)
        
        return predictions, references
    
    def _load_context_test_data(self) -> Tuple[List[str], List[str], List[str]]:
        """Load context-based test data (placeholder)."""
        contexts = [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
            "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can be released to fuel the organism's activities."
        ]
        questions = [
            "Who designed the Eiffel Tower?",
            "What does photosynthesis convert light energy into?"
        ]
        answers = [
            "Gustave Eiffel",
            "chemical energy"
        ]
        return contexts, questions, answers
    
    def _load_satisfaction_prompts(self) -> List[str]:
        """Load prompts for satisfaction evaluation (placeholder)."""
        return [
            "What is artificial intelligence?",
            "Explain the theory of relativity.",
            "How do computers work?",
            "What is the meaning of life?",
            "Write a short poem about nature."
        ]
    
    def _compare_with_benchmarks(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compare results with benchmark models."""
        comparison = {}
        
        if 'perplexity' in self.benchmarks:
            baseline_perplexity = self.benchmarks['perplexity']
            comparison['perplexity_improvement'] = (
                baseline_perplexity - results['perplexity']
            ) / baseline_perplexity
        
        if 'bleu' in self.benchmarks:
            baseline_bleu = self.benchmarks['bleu']
            comparison['bleu_improvement'] = (
                results['bleu']['bleu_with_bp'] - baseline_bleu
            ) / baseline_bleu
        
        if 'rouge' in self.benchmarks:
            baseline_rouge = self.benchmarks['rouge']
            comparison['rouge_improvement'] = (
                results['rouge']['rouge_l_f1'] - baseline_rouge
            ) / baseline_rouge
        
        return comparison
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """Print formatted evaluation report."""
        print("\n" + "="*60)
        print("TmpAi Standard 1.0 - Evaluation Report")
        print("="*60)
        
        print(f"\nTimestamp: {results['timestamp']}")
        print(f"\nModel Configuration:")
        for key, value in results['model_config'].items():
            print(f"  {key}: {value}")
        
        print(f"\nCore Metrics:")
        print(f"  Perplexity: {results['perplexity']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        
        print(f"\nGeneration Quality:")
        print(f"  BLEU: {results['bleu']['bleu_with_bp']:.4f}")
        print(f"  ROUGE-L: {results['rouge']['rouge_l_f1']:.4f}")
        
        print(f"\nContext Retention:")
        print(f"  Accuracy: {results['context_retention']['context_accuracy']:.4f}")
        print(f"  Relevance: {results['context_retention']['context_relevance']:.4f}")
        
        print(f"\nUser Satisfaction:")
        print(f"  Relevance: {results['user_satisfaction']['relevance']:.4f}")
        print(f"  Coherence: {results['user_satisfaction']['coherence']:.4f}")
        print(f"  Helpfulness: {results['user_satisfaction']['helpfulness']:.4f}")
        print(f"  Safety: {results['user_satisfaction']['safety']:.4f}")
        
        if results['benchmark_comparison']:
            print(f"\nBenchmark Comparison (Improvement vs baseline):")
            for metric, improvement in results['benchmark_comparison'].items():
                print(f"  {metric}: {improvement*100:.2f}%")
        
        print("="*60 + "\n")
