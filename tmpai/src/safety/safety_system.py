"""
TmpAi Standard 1.0 - Ethics and Safety Module
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import re
from pathlib import Path
import json
from collections import defaultdict

from tmpai.models import TmpAiModel


@dataclass
class SafetyViolation:
    """Represents a detected safety violation."""
    violation_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    detected_content: str
    explanation: str
    confidence: float


class BiasAnalyzer:
    """Analyzes and mitigates bias in model outputs."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'config/bias_config.json'
        self.load_config()
    
    def load_config(self):
        """Load bias detection configuration."""
        default_config = {
            'protected_attributes': [
                'gender', 'race', 'religion', 'age', 'disability',
                'sexual_orientation', 'nationality', 'ethnicity'
            ],
            'stereotypical_phrases': [],
            'gendered_words': {
                'masculine': ['he', 'him', 'his', 'man', 'men', 'father', 'brother'],
                'feminine': ['she', 'her', 'hers', 'woman', 'women', 'mother', 'sister']
            }
        }
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = default_config
    
    def analyze_bias(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for potential biases.
        
        Returns:
            Dictionary containing bias analysis results
        """
        results = {
            'has_bias': False,
            'bias_types': [],
            'gender_balance': {},
            'protected_attribute_mentions': [],
            'overall_score': 1.0  # 1.0 = no bias, 0.0 = high bias
        }
        
        # Check gender balance
        gender_balance = self._check_gender_balance(text)
        results['gender_balance'] = gender_balance
        
        # Check for protected attribute mentions
        protected_mentions = self._check_protected_attributes(text)
        results['protected_attribute_mentions'] = protected_mentions
        
        # Calculate overall bias score
        if gender_balance['imbalance_score'] < 0.7:
            results['bias_types'].append('gender_imbalance')
            results['has_bias'] = True
            results['overall_score'] *= gender_balance['imbalance_score']
        
        if protected_mentions:
            results['bias_types'].append('protected_attribute_bias')
            results['has_bias'] = True
            results['overall_score'] *= 0.8
        
        return results
    
    def _check_gender_balance(self, text: str) -> Dict[str, float]:
        """Check gender representation balance in text."""
        text_lower = text.lower()
        
        masculine_count = sum(1 for word in self.config['gendered_words']['masculine'] 
                             if word in text_lower.split())
        feminine_count = sum(1 for word in self.config['gendered_words']['feminine'] 
                           if word in text_lower.split())
        
        total = masculine_count + feminine_count
        
        if total == 0:
            return {
                'masculine_count': 0,
                'feminine_count': 0,
                'balance_ratio': 1.0,
                'imbalance_score': 1.0
            }
        
        balance_ratio = min(masculine_count, feminine_count) / total
        imbalance_score = max(balance_ratio, 1.0 - abs(0.5 - balance_ratio) * 2)
        
        return {
            'masculine_count': masculine_count,
            'feminine_count': feminine_count,
            'balance_ratio': balance_ratio,
            'imbalance_score': imbalance_score
        }
    
    def _check_protected_attributes(self, text: str) -> List[Dict[str, str]]:
        """Check mentions of protected attributes in potentially biased contexts."""
        mentions = []
        
        # This is a simplified check - in production, use more sophisticated NLP
        text_lower = text.lower()
        words = text_lower.split()
        
        for attribute in self.config['protected_attributes']:
            # Check if attribute is mentioned (simplified)
            if attribute in text_lower:
                mentions.append({
                    'attribute': attribute,
                    'context': text[max(0, text_lower.find(attribute) - 20):text_lower.find(attribute) + 20]
                })
        
        return mentions
    
    def suggest_improvements(self, text: str) -> List[str]:
        """Suggest improvements to reduce bias."""
        suggestions = []
        bias_analysis = self.analyze_bias(text)
        
        if 'gender_imbalance' in bias_analysis['bias_types']:
            suggestions.append(
                "Consider using gender-neutral language (e.g., 'they' instead of 'he/she')"
            )
            suggestions.append(
                "Ensure balanced representation across genders in examples"
            )
        
        if 'protected_attribute_bias' in bias_analysis['bias_types']:
            suggestions.append(
                "Review mentions of protected attributes for potential stereotyping"
            )
            suggestions.append(
                "Ensure fair treatment across different demographic groups"
            )
        
        return suggestions


class ContentFilter:
    """Filters potentially harmful or inappropriate content."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'config/content_filter_config.json'
        self.load_config()
    
    def load_config(self):
        """Load content filtering configuration."""
        default_config = {
            'blocked_categories': {
                'violence': {
                    'severity': 'high',
                    'keywords': ['kill', 'murder', 'assault', 'violence', 'weapon'],
                    'patterns': []
                },
                'hate_speech': {
                    'severity': 'critical',
                    'keywords': ['hate', 'discrimination', 'slur'],
                    'patterns': []
                },
                'self_harm': {
                    'severity': 'critical',
                    'keywords': ['suicide', 'self-harm', 'kill myself'],
                    'patterns': []
                },
                'sexual_content': {
                    'severity': 'medium',
                    'keywords': ['explicit', 'pornographic'],
                    'patterns': []
                },
                'illegal_activities': {
                    'severity': 'high',
                    'keywords': ['illegal', 'crime', 'fraud', 'hack', 'exploit'],
                    'patterns': []
                }
            },
            'allowed_contexts': {
                'violence': ['history', 'fiction', 'news', 'educational'],
                'illegal_activities': ['legal', 'educational', 'prevention']
            }
        }
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = default_config
    
    def filter_content(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Tuple[bool, List[SafetyViolation]]:
        """
        Filter content for violations.
        
        Returns:
            Tuple of (is_safe, list of violations)
        """
        violations = []
        text_lower = text.lower()
        
        for category, config in self.config['blocked_categories'].items():
            # Check keywords
            for keyword in config['keywords']:
                if keyword in text_lower:
                    # Check if it's in an allowed context
                    if context and self._is_allowed_context(category, context):
                        continue
                    
                    violation = SafetyViolation(
                        violation_type=category,
                        severity=config['severity'],
                        detected_content=keyword,
                        explanation=f"Detected {category} content: '{keyword}'",
                        confidence=0.8
                    )
                    violations.append(violation)
        
        # Additional pattern-based checks
        violations.extend(self._check_patterns(text))
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def _is_allowed_context(self, category: str, context: str) -> bool:
        """Check if content is in an allowed context for the category."""
        if category not in self.config['allowed_contexts']:
            return False
        
        allowed = self.config['allowed_contexts'][category]
        context_lower = context.lower()
        
        return any(allowed_type in context_lower for allowed_type in allowed)
    
    def _check_patterns(self, text: str) -> List[SafetyViolation]:
        """Check text for problematic patterns."""
        violations = []
        
        # Pattern for personal information (email, phone, SSN)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        
        if re.search(email_pattern, text):
            violations.append(SafetyViolation(
                violation_type='personal_info',
                severity='medium',
                detected_content=re.search(email_pattern, text).group(),
                explanation="Detected email address",
                confidence=0.9
            ))
        
        if re.search(phone_pattern, text):
            violations.append(SafetyViolation(
                violation_type='personal_info',
                severity='medium',
                detected_content=re.search(phone_pattern, text).group(),
                explanation="Detected phone number",
                confidence=0.7
            ))
        
        if re.search(ssn_pattern, text):
            violations.append(SafetyViolation(
                violation_type='personal_info',
                severity='high',
                detected_content=re.search(ssn_pattern, text).group(),
                explanation="Detected potential SSN",
                confidence=0.85
            ))
        
        return violations


class SafetySystem:
    """
    Comprehensive safety system for TmpAi Standard 1.0
    
    Integrates bias analysis, content filtering, and safety guidelines
    to ensure ethical AI usage.
    """
    
    def __init__(
        self,
        model: TmpAiModel,
        config_path: Optional[str] = None
    ):
        self.model = model
        self.config_path = config_path or 'config/safety_config.json'
        self.load_config()
        
        self.bias_analyzer = BiasAnalyzer()
        self.content_filter = ContentFilter()
        
        self.violation_log: List[Dict[str, Any]] = []
    
    def load_config(self):
        """Load safety system configuration."""
        default_config = {
            'enable_bias_detection': True,
            'enable_content_filtering': True,
            'enable_redaction': True,
            'max_violations_per_session': 10,
            'auto_refusal_threshold': 'critical',
            'warning_threshold': 'high'
        }
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = default_config
    
    def check_input(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check input for safety violations.
        
        Returns:
            Dictionary with check results
        """
        result = {
            'is_safe': True,
            'violations': [],
            'bias_analysis': None,
            'should_refuse': False
        }
        
        # Content filtering
        if self.config['enable_content_filtering']:
            is_safe, violations = self.content_filter.filter_content(text, context)
            result['is_safe'] = result['is_safe'] and is_safe
            result['violations'].extend(violations)
        
        # Bias analysis
        if self.config['enable_bias_detection']:
            bias_analysis = self.bias_analyzer.analyze_bias(text)
            result['bias_analysis'] = bias_analysis
            if bias_analysis['has_bias'] and bias_analysis['overall_score'] < 0.5:
                result['is_safe'] = False
        
        # Determine if should refuse
        if violations:
            severities = [v.severity for v in violations]
            if self._should_refuse(severities):
                result['should_refuse'] = True
        
        # Log violations
        for violation in violations:
            self._log_violation('input', text, violation)
        
        return result
    
    def check_output(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check model output for safety violations."""
        return self.check_input(text, context)
    
    def redact_content(self, text: str) -> Tuple[str, List[str]]:
        """
        Redact sensitive content from text.
        
        Returns:
            Tuple of (redacted_text, list_of_redacted_items)
        """
        redacted_text = text
        redacted_items = []
        
        if not self.config['enable_redaction']:
            return redacted_text, redacted_items
        
        # Redact personal information
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        
        for pattern, name in [(email_pattern, 'EMAIL'), (phone_pattern, 'PHONE'), (ssn_pattern, 'SSN')]:
            matches = re.finditer(pattern, redacted_text)
            for match in reversed(list(matches)):
                redacted = f'[{name}]'
                redacted_text = redacted_text[:match.start()] + redacted + redacted_text[match.end():]
                redacted_items.append(f"{name}: {match.group()}")
        
        return redacted_text, redacted_items
    
    def generate_refusal_message(self, violation_type: str) -> str:
        """Generate appropriate refusal message based on violation type."""
        refusal_messages = {
            'violence': "I cannot assist with requests related to violence or harmful activities.",
            'hate_speech': "I cannot generate content that promotes hate speech or discrimination.",
            'self_harm': "If you're experiencing thoughts of self-harm, please reach out for help. "
                         "You can contact emergency services or crisis hotlines for support.",
            'sexual_content': "I cannot generate explicit sexual content.",
            'illegal_activities': "I cannot assist with requests for illegal activities.",
            'personal_info': "Personal information has been redacted for privacy protection."
        }
        
        return refusal_messages.get(
            violation_type,
            "I apologize, but I cannot fulfill this request as it may violate safety guidelines."
        )
    
    def _should_refuse(self, severities: List[str]) -> bool:
        """Determine if should refuse based on violation severities."""
        severity_order = ['low', 'medium', 'high', 'critical']
        threshold = severity_order.index(self.config['auto_refusal_threshold'])
        
        for severity in severities:
            if severity_order.index(severity) >= threshold:
                return True
        
        return False
    
    def _log_violation(
        self,
        stage: str,
        content: str,
        violation: SafetyViolation
    ) -> None:
        """Log safety violation."""
        from datetime import datetime
        
        self.violation_log.append({
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'violation_type': violation.violation_type,
            'severity': violation.severity,
            'explanation': violation.explanation,
            'confidence': violation.confidence,
            'content_snippet': content[:100]  # Store only snippet
        })
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate safety report."""
        if not self.violation_log:
            return {'total_violations': 0}
        
        violation_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for log in self.violation_log:
            violation_counts[log['violation_type']] += 1
            severity_counts[log['severity']] += 1
        
        return {
            'total_violations': len(self.violation_log),
            'by_type': dict(violation_counts),
            'by_severity': dict(severity_counts)
        }
    
    def save_safety_report(self, filepath: str) -> None:
        """Save safety report to file."""
        report = {
            'safety_report': self.get_safety_report(),
            'violation_log': self.violation_log[-100:]  # Last 100 violations
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


class SafetyGuidelines:
    """Documentation of safety and ethical guidelines."""
    
    GUIDELINES = {
        'fairness': [
            "Treat all users with respect and without bias",
            "Avoid stereotypes based on protected characteristics",
            "Provide balanced and inclusive representations"
        ],
        'transparency': [
            "Be clear about the model's capabilities and limitations",
            "Acknowledge uncertainty when appropriate",
            "Do not claim human-like consciousness or agency"
        ],
        'privacy': [
            "Do not generate or expose personal information",
            "Redact sensitive data when possible",
            "Respect user privacy at all times"
        ],
        'safety': [
            "Do not generate harmful content",
            "Refuse requests for illegal activities",
            "Provide resources for crisis situations"
        ],
        'accountability': [
            "Take responsibility for model outputs",
            "Implement feedback mechanisms",
            "Continuously improve based on user feedback"
        ]
    }
    
    @classmethod
    def get_guideline_documentation(cls) -> str:
        """Get formatted guideline documentation."""
        sections = []
        
        for category, guidelines in cls.GUIDELINES.items():
            section = f"\n{category.upper()}\n"
            section += "=" * len(category) + "\n"
            for i, guideline in enumerate(guidelines, 1):
                section += f"{i}. {guideline}\n"
            sections.append(section)
        
        return "\n".join(sections)
    
    @classmethod
    def save_guidelines(cls, filepath: str) -> None:
        """Save guidelines to file."""
        with open(filepath, 'w') as f:
            f.write("TmpAi Standard 1.0 - Ethical Guidelines\n")
            f.write("=" * 50 + "\n\n")
            f.write(cls.get_guideline_documentation())
