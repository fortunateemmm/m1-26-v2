"""
Reward functions for GRPO training.
Implements multi-dimensional rewards for medical reasoning:
- Answer correctness
- Format compliance
- Semantic quality
"""

import re
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from model output.
    
    Supports multiple formats:
    - \boxed{answer}
    - Answer: answer
    - The answer is answer
    """
    # Try \boxed{} format first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Try "Answer:" format
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Try "The answer is" format
    is_match = re.search(r'the answer is\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if is_match:
        return is_match.group(1).strip()
    
    # Try to find letter choice (A, B, C, D)
    choice_match = re.search(r'\b([A-D])\b(?:\.|\)|\s+is)', text)
    if choice_match:
        return choice_match.group(1)
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    # Remove extra whitespace
    answer = answer.strip()
    
    # Remove trailing punctuation
    answer = answer.rstrip('.:,;')
    
    # Lowercase
    answer = answer.lower()
    
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', '', answer)
    
    # Remove extra spaces
    answer = ' '.join(answer.split())
    
    return answer


def compute_answer_reward(
    response: str,
    ground_truth: str,
) -> float:
    """
    Compute answer correctness reward.
    
    Args:
        response: Model's generated response
        ground_truth: Correct answer
        
    Returns:
        Reward score (0.0 or 1.0)
    """
    predicted = extract_answer(response)
    
    if predicted is None:
        return 0.0
    
    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Exact match
    if pred_norm == gt_norm:
        return 1.0
    
    # Check if prediction contains ground truth (for longer answers)
    if gt_norm in pred_norm:
        return 0.8
    
    # Check if ground truth contains prediction
    if pred_norm in gt_norm and len(pred_norm) > 3:
        return 0.5
    
    return 0.0


def compute_format_reward(response: str) -> float:
    """
    Compute format compliance reward.
    
    Checks for:
    - Presence of \boxed{} format
    - Reasoning structure (think/answer tags)
    - Appropriate length
    
    Returns:
        Reward score (0.0 to 1.0)
    """
    score = 0.0
    max_score = 3.0
    
    # Check for \boxed{} format
    if re.search(r'\\boxed\{[^}]+\}', response):
        score += 1.0
    
    # Check for reasoning structure
    has_think = bool(re.search(r'<think>|thinking|reasoning|step', response, re.IGNORECASE))
    if has_think:
        score += 1.0
    
    # Check for reasonable length (not too short, not too long)
    word_count = len(response.split())
    if 50 <= word_count <= 2000:
        score += 1.0
    elif word_count > 20:
        score += 0.5
    
    return score / max_score


def compute_semantic_reward(
    response: str,
    question: str,
) -> float:
    """
    Compute semantic quality reward using heuristic rules.
    
    Checks for:
    - Coherence (no repetitive patterns)
    - Relevance to medical domain
    - Logical flow
    
    Args:
        response: Model's generated response
        question: Original question
        
    Returns:
        Reward score (0.0 to 1.0)
    """
    score = 0.0
    max_score = 4.0
    
    # 1. Check for repetitive patterns (penalize)
    words = response.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.4:
            score += 1.0
        elif unique_ratio > 0.25:
            score += 0.5
    else:
        score += 0.5  # Short responses get benefit of doubt
    
    # 2. Check for medical terminology presence
    medical_terms = [
        'patient', 'diagnosis', 'treatment', 'symptom', 'disease',
        'clinical', 'medical', 'health', 'therapy', 'condition',
        'analysis', 'evidence', 'study', 'research', 'data',
        'blood', 'heart', 'lung', 'brain', 'liver', 'kidney',
        'infection', 'inflammation', 'chronic', 'acute',
    ]
    term_count = sum(1 for term in medical_terms if term in response.lower())
    if term_count >= 3:
        score += 1.0
    elif term_count >= 1:
        score += 0.5
    
    # 3. Check for logical connectors (indicates reasoning)
    logical_connectors = [
        'therefore', 'because', 'since', 'however', 'moreover',
        'furthermore', 'consequently', 'thus', 'hence', 'additionally',
        'first', 'second', 'third', 'finally', 'in conclusion',
        'this suggests', 'this indicates', 'based on',
    ]
    connector_count = sum(1 for conn in logical_connectors if conn in response.lower())
    if connector_count >= 2:
        score += 1.0
    elif connector_count >= 1:
        score += 0.5
    
    # 4. Check response coherence (no sudden topic jumps)
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) >= 3:
        score += 1.0
    elif len(sentences) >= 1:
        score += 0.5
    
    return score / max_score


def compute_comprehensive_semantic_reward(
    response: str,
    question: str,
    ground_truth: str,
) -> float:
    """
    Compute comprehensive semantic reward combining multiple signals.
    
    This is a more sophisticated version that also considers:
    - Answer explanation quality
    - Consistency between reasoning and answer
    
    Args:
        response: Model's generated response
        question: Original question
        ground_truth: Correct answer
        
    Returns:
        Reward score (0.0 to 1.0)
    """
    base_semantic = compute_semantic_reward(response, question)
    
    # Additional checks for answer-reasoning consistency
    consistency_score = 0.0
    max_consistency = 2.0
    
    # Check if answer appears in response context (not just at the end)
    answer = extract_answer(response)
    if answer:
        answer_lower = answer.lower()
        response_lower = response.lower()
        
        # Find all occurrences of answer
        occurrences = response_lower.count(answer_lower)
        if occurrences >= 2:  # Answer mentioned multiple times
            consistency_score += 1.0
        
        # Check if there's explanation before the final answer
        boxed_pos = response_lower.find('\\boxed{')
        if boxed_pos > 100:  # Reasonable amount of text before answer
            consistency_score += 1.0
        elif boxed_pos == -1 and len(response) > 100:
            consistency_score += 0.5
    
    consistency = consistency_score / max_consistency
    
    # Combine scores
    return 0.7 * base_semantic + 0.3 * consistency


def compute_combined_reward(
    response: str,
    ground_truth: str,
    question: str,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute combined reward from all reward components.
    
    Args:
        response: Model's generated response
        ground_truth: Correct answer
        question: Original question
        weights: Reward component weights
        
    Returns:
        Dictionary with individual and total rewards
    """
    if weights is None:
        weights = {
            "answer": 1.0,
            "format": 0.2,
            "semantic": 0.3,
        }
    
    # Compute individual rewards
    answer_reward = compute_answer_reward(response, ground_truth)
    format_reward = compute_format_reward(response)
    semantic_reward = compute_comprehensive_semantic_reward(
        response, question, ground_truth
    )
    
    # Weighted combination
    total = (
        weights["answer"] * answer_reward
        + weights["format"] * format_reward
        + weights["semantic"] * semantic_reward
    )
    
    return {
        "total": total,
        "answer": answer_reward,
        "format": format_reward,
        "semantic": semantic_reward,
    }


class RewardComputer:
    """
    Stateful reward computer that can cache computations
    and provide additional context.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_llm_judge: bool = False,
    ):
        self.weights = weights or {
            "answer": 1.0,
            "format": 0.2,
            "semantic": 0.3,
        }
        self.use_llm_judge = use_llm_judge
        self._cache = {}
    
    def __call__(
        self,
        response: str,
        ground_truth: str,
        question: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute reward with optional caching."""
        cache_key = hash((response, ground_truth, question))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = compute_combined_reward(
            response, ground_truth, question,
            weights or self.weights,
        )
        
        self._cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Clear the reward cache."""
        self._cache.clear()
