"""
数据预处理脚本

只负责质量验证，输出一个统一的数据集
训练时会根据per-sample loss动态选择SFT或GRPO
"""

import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataProcessingConfig:
    """数据处理配置"""
    input_path: str
    output_path: str
    min_format_score: float = 0.6
    min_coherence_score: float = 0.5
    min_consistency_score: float = 0.6
    min_total_score: float = 0.55
    min_difficulty: float = 0.3  # 过滤太简单的样本


class ReasoningQualityVerifier:
    """推理质量验证器"""
    
    MEDICAL_TERMS = [
        'diagnosis', 'treatment', 'symptom', 'pathology', 'prognosis',
        'etiology', 'differential', 'contraindication', 'indication',
        'benign', 'malignant', 'acute', 'chronic', 'systemic',
    ]
    
    LOGICAL_CONNECTORS = [
        'therefore', 'because', 'since', 'thus', 'hence',
        'consequently', 'however', 'moreover', 'first', 'second', 'finally',
    ]
    
    def verify_sample(self, sample: Dict) -> Dict:
        """验证单个样本"""
        question = sample.get("question", "")
        reasoning = sample.get("reasoning", "")
        answer = sample.get("answer", sample.get("distilled_answer_string", ""))
        
        scores = {
            "format_score": self._score_format(reasoning),
            "coherence_score": self._score_coherence(reasoning),
            "consistency_score": self._score_consistency(reasoning, answer),
            "medical_score": self._score_medical(reasoning),
        }
        
        weights = {"format_score": 0.20, "coherence_score": 0.25, 
                   "consistency_score": 0.35, "medical_score": 0.20}
        scores["total_score"] = sum(scores[k] * weights[k] for k in weights)
        
        return {**sample, **scores}
    
    def compute_difficulty(self, sample: Dict) -> float:
        """计算样本难度 (0-1)"""
        score = 0.0
        
        # 基于推理长度
        reasoning = sample.get("reasoning", "")
        word_count = len(reasoning.split())
        if word_count > 500:
            score += 0.3
        elif word_count > 200:
            score += 0.2
        else:
            score += 0.1
        
        # 基于步骤数量
        step_count = sum(1 for s in ['step 1', 'first,', 'second,'] if s in reasoning.lower())
        if step_count >= 3:
            score += 0.3
        elif step_count >= 1:
            score += 0.2
        else:
            score += 0.1
        
        # 基于问题类型（选择题 vs 开放式）
        if any(x in sample.get("question", "") for x in ['A.', 'B.', 'C.', 'D.']):
            score += 0.2  # 选择题相对简单
        else:
            score += 0.4  # 开放式更难
        
        # 基于医学术语数量
        term_count = sum(1 for t in self.MEDICAL_TERMS if t in reasoning.lower())
        if term_count >= 3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_format(self, reasoning: str) -> float:
        if not reasoning:
            return 0.0
        score = 0.0
        if any(p in reasoning.lower() for p in ['step 1', 'first,']):
            score += 0.25
        if any(p in reasoning.lower() for p in ['therefore', 'conclusion', 'thus']):
            score += 0.25
        word_count = len(reasoning.split())
        if 50 <= word_count <= 1000:
            score += 0.25
        if any(p in reasoning.lower() for p in ['according to', 'studies show']):
            score += 0.25
        return score
    
    def _score_coherence(self, reasoning: str) -> float:
        if not reasoning:
            return 0.0
        score = 0.0
        connector_count = sum(1 for c in self.LOGICAL_CONNECTORS if c in reasoning.lower())
        score += min(connector_count / 5, 1.0) * 0.4
        sentences = [s for s in reasoning.split('.') if len(s.strip()) > 10]
        score += min(len(sentences) / 10, 1.0) * 0.3
        words = reasoning.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.3
        return score
    
    def _score_consistency(self, reasoning: str, answer: str) -> float:
        if not reasoning or not answer:
            return 0.5
        score = 0.0
        answer_lower = answer.lower().strip()
        reasoning_lower = reasoning.lower()
        if answer_lower in reasoning_lower:
            score += 0.5
        elif any(w in reasoning_lower for w in answer_lower.split() if len(w) > 3):
            score += 0.3
        import re
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', reasoning)
        if boxed_match and boxed_match.group(1).strip().lower() == answer_lower:
            score += 0.5
        return min(score, 1.0)
    
    def _score_medical(self, reasoning: str) -> float:
        if not reasoning:
            return 0.0
        term_count = sum(1 for t in self.MEDICAL_TERMS if t in reasoning.lower())
        return min(term_count / 5, 1.0)


def process_data(config: DataProcessingConfig):
    """
    处理数据：质量验证 + 难度计算
    
    输出一个统一的数据集，训练时会根据难度动态选择SFT或GRPO
    """
    print(f"\n{'='*60}")
    print(f"Data Processing Pipeline")
    print(f"{'='*60}")
    
    # 加载数据
    print(f"\n[1] Loading data from {config.input_path}...")
    samples = []
    with open(config.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"    Loaded {len(samples)} samples")
    
    # 质量验证和难度计算
    print(f"\n[2] Verifying quality and computing difficulty...")
    verifier = ReasoningQualityVerifier()
    
    processed_samples = []
    stats = {"total": 0, "passed": 0, "failed_quality": 0, "failed_difficulty": 0}
    
    for sample in samples:
        stats["total"] += 1
        
        # 验证质量
        verified = verifier.verify_sample(sample)
        
        if verified["total_score"] < config.min_total_score:
            stats["failed_quality"] += 1
            continue
        
        # 计算难度
        difficulty = verifier.compute_difficulty(verified)
        
        if difficulty < config.min_difficulty:
            stats["failed_difficulty"] += 1
            continue
        
        # 添加难度信息
        verified["difficulty"] = difficulty
        processed_samples.append(verified)
        stats["passed"] += 1
    
    print(f"    Total: {stats['total']}")
    print(f"    Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"    Failed quality: {stats['failed_quality']}")
    print(f"    Failed difficulty (too easy): {stats['failed_difficulty']}")
    
    # 打印难度分布
    difficulties = [s["difficulty"] for s in processed_samples]
    print(f"\n    Difficulty distribution:")
    print(f"      Easy (0.3-0.5): {sum(1 for d in difficulties if 0.3 <= d < 0.5)}")
    print(f"      Medium (0.5-0.7): {sum(1 for d in difficulties if 0.5 <= d < 0.7)}")
    print(f"      Hard (0.7-1.0): {sum(1 for d in difficulties if 0.7 <= d <= 1.0)}")
    
    # 保存数据
    print(f"\n[3] Saving processed data to {config.output_path}...")
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    
    with open(config.output_path, 'w', encoding='utf-8') as f:
        for sample in processed_samples:
            # 只保留必要字段
            clean_sample = {
                "question": sample.get("question", ""),
                "prompt": sample.get("prompt", sample.get("question", "")),
                "reasoning": sample.get("reasoning", ""),
                "answer": sample.get("answer", sample.get("distilled_answer_string", "")),
                "difficulty": sample.get("difficulty", 0.5),
                "quality": sample.get("total_score", 0.5),
            }
            f.write(json.dumps(clean_sample, ensure_ascii=False) + '\n')
    
    print(f"    Saved {len(processed_samples)} samples")
    print(f"\n{'='*60}")
    print(f"Data processing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process training data")
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument("--output", required=True, help="Output data file")
    parser.add_argument("--min_total", type=float, default=0.55)
    parser.add_argument("--min_difficulty", type=float, default=0.3)
    args = parser.parse_args()
    
    config = DataProcessingConfig(
        input_path=args.input,
        output_path=args.output,
        min_total_score=args.min_total,
        min_difficulty=args.min_difficulty,
    )
    
    process_data(config)
