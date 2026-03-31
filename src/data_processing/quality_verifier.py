"""
数据质量验证器

用于在训练前验证生成的推理（reasoning）质量
支持多维度评估：格式、连贯性、一致性、医学准确性
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """质量阈值配置"""
    min_format_score: float = 0.6
    min_coherence_score: float = 0.5
    min_consistency_score: float = 0.6
    min_medical_score: float = 0.3
    min_total_score: float = 0.55


class ReasoningQualityVerifier:
    """
    推理质量验证器
    
    评估维度：
    1. Format Score - 格式完整性（步骤标记、结论、长度）
    2. Coherence Score - 逻辑连贯性（连接词、句子结构、词汇多样性）
    3. Consistency Score - 答案一致性（推理与最终答案是否一致）
    4. Medical Score - 医学术语使用
    """
    
    MEDICAL_TERMS = [
        'diagnosis', 'treatment', 'symptom', 'pathology', 'prognosis',
        'etiology', 'differential', 'contraindication', 'indication',
        'benign', 'malignant', 'acute', 'chronic', 'systemic',
        'hematologic', 'oncologic', 'cardiologic', 'neurologic',
        'pathophysiology', 'pharmacology', 'therapeutic', 'clinical',
    ]
    
    LOGICAL_CONNECTORS = [
        'therefore', 'because', 'since', 'thus', 'hence',
        'consequently', 'however', 'moreover', 'furthermore',
        'additionally', 'first', 'second', 'third', 'finally',
        'in conclusion', 'based on', 'this suggests', 'this indicates',
    ]
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
    
    def verify_sample(self, sample: Dict) -> Dict:
        """
        验证单个样本的质量
        
        Args:
            sample: 包含 question, reasoning, answer 的字典
            
        Returns:
            包含各维度分数和总分的字典
        """
        question = sample.get("question", "")
        reasoning = sample.get("reasoning", "")
        answer = sample.get("answer", sample.get("distilled_answer_string", ""))
        
        scores = {
            "format_score": self._score_format(reasoning),
            "coherence_score": self._score_coherence(reasoning),
            "consistency_score": self._score_consistency(reasoning, answer),
            "medical_score": self._score_medical(reasoning),
        }
        
        # 计算总分（加权平均）
        weights = {
            "format_score": 0.20,
            "coherence_score": 0.25,
            "consistency_score": 0.35,
            "medical_score": 0.20,
        }
        
        scores["total_score"] = sum(scores[k] * weights[k] for k in weights)
        scores["is_valid"] = self._check_thresholds(scores)
        
        return {**sample, **scores}
    
    def verify_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        验证整个数据集
        
        Returns:
            valid_samples: 通过验证的样本
            statistics: 验证统计信息
        """
        verified_samples = []
        stats = {
            "total": len(dataset),
            "valid": 0,
            "invalid": 0,
            "scores": [],
        }
        
        for sample in dataset:
            verified = self.verify_sample(sample)
            verified_samples.append(verified)
            
            if verified["is_valid"]:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
            
            stats["scores"].append(verified["total_score"])
        
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        stats["pass_rate"] = stats["valid"] / stats["total"] if stats["total"] > 0 else 0
        
        return verified_samples, stats
    
    def filter_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """过滤数据集，只保留通过验证的样本"""
        verified, _ = self.verify_dataset(dataset)
        return [s for s in verified if s["is_valid"]]
    
    def _score_format(self, reasoning: str) -> float:
        """评分：格式完整性"""
        if not reasoning:
            return 0.0
        
        score = 0.0
        
        # 1. 有明确的步骤标记
        step_patterns = ['step 1', 'step 2', 'first,', 'first.', 'second,', 'third,']
        if any(p in reasoning.lower() for p in step_patterns):
            score += 0.25
        
        # 2. 有结论标记
        conclusion_patterns = ['therefore', 'conclusion', 'in summary', 'thus', 'finally']
        if any(p in reasoning.lower() for p in conclusion_patterns):
            score += 0.25
        
        # 3. 长度合理（50-1000词）
        word_count = len(reasoning.split())
        if 50 <= word_count <= 1000:
            score += 0.25
        elif 20 <= word_count < 50 or 1000 < word_count <= 2000:
            score += 0.15
        elif word_count < 20:
            score += 0.05
        
        # 4. 有引用或证据
        evidence_patterns = ['according to', 'studies show', 'research indicates', 'evidence suggests']
        if any(p in reasoning.lower() for p in evidence_patterns):
            score += 0.25
        
        return score
    
    def _score_coherence(self, reasoning: str) -> float:
        """评分：逻辑连贯性"""
        if not reasoning:
            return 0.0
        
        score = 0.0
        
        # 1. 逻辑连接词数量
        connector_count = sum(1 for c in self.LOGICAL_CONNECTORS if c in reasoning.lower())
        if connector_count >= 3:
            score += 0.35
        elif connector_count >= 1:
            score += 0.20
        else:
            score += 0.05
        
        # 2. 句子数量（合理范围）
        sentences = [s.strip() for s in re.split(r'[.!?]+', reasoning) if len(s.strip()) > 10]
        if 3 <= len(sentences) <= 15:
            score += 0.35
        elif 1 <= len(sentences) < 3:
            score += 0.20
        elif len(sentences) > 15:
            score += 0.25
        
        # 3. 词汇多样性
        words = reasoning.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.5:
                score += 0.30
            elif unique_ratio > 0.35:
                score += 0.20
            else:
                score += 0.10
        
        return score
    
    def _score_consistency(self, reasoning: str, answer: str) -> float:
        """评分：推理与答案的一致性"""
        if not reasoning or not answer:
            return 0.5
        
        answer_lower = answer.lower().strip()
        reasoning_lower = reasoning.lower()
        
        score = 0.0
        
        # 1. 答案是否在推理中出现
        if answer_lower in reasoning_lower:
            score += 0.4
        elif len(answer_lower) > 5:
            # 检查部分匹配
            answer_words = set(answer_lower.split())
            reasoning_words = set(reasoning_lower.split())
            overlap = len(answer_words & reasoning_words)
            if overlap >= len(answer_words) * 0.5:
                score += 0.25
            elif overlap > 0:
                score += 0.15
        
        # 2. 检查\boxed中的答案是否与提供的答案一致
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', reasoning)
        if boxed_match:
            boxed_answer = boxed_match.group(1).strip().lower()
            if boxed_answer == answer_lower:
                score += 0.4
            elif answer_lower in boxed_answer or boxed_answer in answer_lower:
                score += 0.2
        
        # 3. 推理结尾是否有明确的结论指向答案
        ending_patterns = [
            r'therefore.*?(?:is|are|should be|would be)\s+' + re.escape(answer_lower[:10]),
            r'the answer is\s+' + re.escape(answer_lower[:10]),
            r'select.*?' + re.escape(answer_lower[:10]),
        ]
        for pattern in ending_patterns:
            if re.search(pattern, reasoning_lower):
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _score_medical(self, reasoning: str) -> float:
        """评分：医学术语使用"""
        if not reasoning:
            return 0.0
        
        reasoning_lower = reasoning.lower()
        term_count = sum(1 for term in self.MEDICAL_TERMS if term in reasoning_lower)
        
        if term_count >= 5:
            return 1.0
        elif term_count >= 3:
            return 0.7
        elif term_count >= 1:
            return 0.4
        else:
            return 0.1
    
    def _check_thresholds(self, scores: Dict) -> bool:
        """检查是否通过所有阈值"""
        return (
            scores["format_score"] >= self.thresholds.min_format_score and
            scores["coherence_score"] >= self.thresholds.min_coherence_score and
            scores["consistency_score"] >= self.thresholds.min_consistency_score and
            scores["medical_score"] >= self.thresholds.min_medical_score and
            scores["total_score"] >= self.thresholds.min_total_score
        )


def verify_data_command(
    input_path: str,
    output_path: str,
    thresholds: Optional[Dict] = None,
):
    """
    命令行入口：验证数据质量
    
    Usage:
        python -m data_processing.quality_verifier \
            --input data/raw_samples.jsonl \
            --output data/verified_samples.jsonl
    """
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify reasoning quality")
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument("--output", required=True, help="Output verified data file")
    parser.add_argument("--min_format", type=float, default=0.6)
    parser.add_argument("--min_coherence", type=float, default=0.5)
    parser.add_argument("--min_consistency", type=float, default=0.6)
    parser.add_argument("--min_total", type=float, default=0.55)
    args = parser.parse_args()
    
    # 加载数据
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    # 设置阈值
    quality_thresholds = QualityThresholds(
        min_format_score=args.min_format,
        min_coherence_score=args.min_coherence,
        min_consistency_score=args.min_consistency,
        min_total_score=args.min_total,
    )
    
    # 验证
    verifier = ReasoningQualityVerifier(quality_thresholds)
    verified_samples, stats = verifier.verify_dataset(samples)
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in verified_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 打印统计
    print(f"\n{'='*50}")
    print(f"Quality Verification Results")
    print(f"{'='*50}")
    print(f"Total samples: {stats['total']}")
    print(f"Valid samples: {stats['valid']} ({stats['pass_rate']*100:.1f}%)")
    print(f"Invalid samples: {stats['invalid']}")
    print(f"Average score: {stats['avg_score']:.3f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    verify_data_command()
