"""
难度感知采样器

根据样本难度和质量进行分层采样
将数据分为：GRPO训练集、SFT训练集、待改进数据
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """采样配置"""
    total_samples: int = 1000
    grpo_ratio: float = 0.5  # GRPO训练集比例
    difficulty_bins: int = 5
    min_quality: float = 0.6
    balance_across_difficulty: bool = True


class DifficultyCalculator:
    """
    难度计算器
    
    基于多个因素计算样本难度：
    1. 原始模型是否能解决
    2. 推理长度和复杂度
    3. 问题类型
    """
    
    def compute_difficulty(self, sample: Dict) -> float:
        """
        计算样本难度分数 (0-1)
        
        Args:
            sample: 包含各种元数据的样本
            
        Returns:
            难度分数，越高越难
        """
        score = 0.0
        
        # 1. 基于原始模型是否解决 (0-0.5)
        solved_by_base = sample.get("solved_by_base", False)
        if not solved_by_base:
            score += 0.5  # 困难：基础模型无法解决
        else:
            score += 0.0  # 简单：基础模型能解决
        
        # 2. 基于推理复杂度 (0-0.3)
        reasoning = sample.get("reasoning", "")
        word_count = len(reasoning.split())
        
        if word_count > 500:
            score += 0.15
        elif word_count > 200:
            score += 0.10
        else:
            score += 0.05
        
        # 基于步骤数量
        step_indicators = ['step 1', 'step 2', 'step 3', 'first,', 'second,', 'third,']
        step_count = sum(1 for indicator in step_indicators if indicator in reasoning.lower())
        
        if step_count >= 4:
            score += 0.15
        elif step_count >= 2:
            score += 0.10
        else:
            score += 0.05
        
        # 3. 基于问题类型 (0-0.2)
        question = sample.get("question", "")
        if self._is_multiple_choice(question):
            score += 0.05  # 选择题相对简单
        else:
            score += 0.15  # 开放式问题更难
        
        return min(score, 1.0)
    
    def _is_multiple_choice(self, question: str) -> bool:
        """判断是否为选择题"""
        # 检查是否有A. B. C. D.选项
        choices = ['A.', 'B.', 'C.', 'D.', 'A)', 'B)', 'C)', 'D)']
        return sum(1 for c in choices if c in question) >= 3
    
    def bin_difficulty(self, difficulty: float, n_bins: int = 5) -> int:
        """将难度分数分桶"""
        return min(int(difficulty * n_bins), n_bins - 1)


class DifficultyAwareSampler:
    """
    难度感知采样器
    
    根据难度和质量进行分层采样
    """
    
    def __init__(
        self,
        quality_scorer=None,
        config: Optional[SamplingConfig] = None,
    ):
        from .quality_verifier import ReasoningQualityVerifier
        
        self.quality_scorer = quality_scorer or ReasoningQualityVerifier()
        self.config = config or SamplingConfig()
        self.difficulty_calculator = DifficultyCalculator()
    
    def categorize_samples(
        self,
        samples: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """
        将样本分类为不同难度和质量等级
        
        Returns:
            {
                "hard_high_quality": [...],   # GRPO训练集
                "hard_low_quality": [...],    # 需要重新生成
                "medium_high_quality": [...], # SFT训练集补充
                "easy_high_quality": [...],   # SFT训练集
                "easy_low_quality": [...],    # 需要改进
            }
        """
        categorized = {
            "hard_high_quality": [],
            "hard_low_quality": [],
            "medium_high_quality": [],
            "easy_high_quality": [],
            "easy_low_quality": [],
        }
        
        for sample in samples:
            # 计算难度
            difficulty = self.difficulty_calculator.compute_difficulty(sample)
            
            # 计算质量（如果没有预先计算）
            if "total_score" not in sample:
                verified = self.quality_scorer.verify_sample(sample)
                quality = verified["total_score"]
            else:
                quality = sample["total_score"]
            
            sample["difficulty"] = difficulty
            sample["quality"] = quality
            
            # 分类
            is_hard = difficulty >= 0.6
            is_medium = 0.3 <= difficulty < 0.6
            is_high_quality = quality >= self.config.min_quality
            
            if is_hard and is_high_quality:
                category = "hard_high_quality"
            elif is_hard and not is_high_quality:
                category = "hard_low_quality"
            elif is_medium and is_high_quality:
                category = "medium_high_quality"
            elif not is_hard and is_high_quality:
                category = "easy_high_quality"
            else:
                category = "easy_low_quality"
            
            categorized[category].append(sample)
        
        # 打印统计
        self._print_category_stats(categorized)
        
        return categorized
    
    def sample_training_sets(
        self,
        categorized: Dict[str, List[Dict]],
    ) -> Dict[str, List[Dict]]:
        """
        从分类数据中采样训练集
        
        Returns:
            {
                "grpo_set": [...],  # 困难+高质量，用于GRPO训练
                "sft_set": [...],   # 简单+高质量，用于SFT训练
                "rejection_set": [...],  # 低质量，需要改进
            }
        """
        n_grpo = int(self.config.total_samples * self.config.grpo_ratio)
        n_sft = self.config.total_samples - n_grpo
        
        # GRPO集：从hard_high_quality中采样
        grpo_candidates = categorized["hard_high_quality"]
        grpo_set = self._sample_by_difficulty(
            grpo_candidates, n_grpo, self.config.difficulty_bins
        ) if self.config.balance_across_difficulty else grpo_candidates[:n_grpo]
        
        # SFT集：从easy_high_quality + medium_high_quality中采样
        sft_candidates = categorized["easy_high_quality"] + categorized["medium_high_quality"]
        sft_set = self._sample_by_difficulty(
            sft_candidates, n_sft, self.config.difficulty_bins
        ) if self.config.balance_across_difficulty else sft_candidates[:n_sft]
        
        # 拒绝集：所有低质量样本
        rejection_set = (
            categorized["hard_low_quality"] +
            categorized["easy_low_quality"]
        )
        
        print(f"\n{'='*50}")
        print(f"Training Set Statistics")
        print(f"{'='*50}")
        print(f"GRPO set (hard+high quality): {len(grpo_set)} samples")
        print(f"SFT set (easy/medium+high quality): {len(sft_set)} samples")
        print(f"Rejection set (low quality): {len(rejection_set)} samples")
        print(f"{'='*50}\n")
        
        return {
            "grpo_set": grpo_set,
            "sft_set": sft_set,
            "rejection_set": rejection_set,
        }
    
    def _sample_by_difficulty(
        self,
        samples: List[Dict],
        n_samples: int,
        n_bins: int,
    ) -> List[Dict]:
        """按难度分层采样"""
        if len(samples) <= n_samples:
            return samples
        
        # 按难度分桶
        bins = defaultdict(list)
        for sample in samples:
            difficulty_bin = self.difficulty_calculator.bin_difficulty(
                sample["difficulty"], n_bins
            )
            bins[difficulty_bin].append(sample)
        
        # 每个桶采样相同数量
        samples_per_bin = n_samples // n_bins
        result = []
        
        for bin_id in range(n_bins):
            bin_samples = bins.get(bin_id, [])
            # 按质量排序
            bin_samples.sort(key=lambda x: x.get("quality", 0), reverse=True)
            # 采样
            result.extend(bin_samples[:samples_per_bin])
        
        # 如果还有剩余配额，按质量补充
        if len(result) < n_samples:
            remaining = sorted(
                [s for s in samples if s not in result],
                key=lambda x: x.get("quality", 0),
                reverse=True
            )
            result.extend(remaining[:n_samples - len(result)])
        
        return result
    
    def _print_category_stats(self, categorized: Dict[str, List[Dict]]):
        """打印分类统计"""
        print(f"\n{'='*50}")
        print(f"Sample Categorization")
        print(f"{'='*50}")
        
        total = sum(len(v) for v in categorized.values())
        for category, samples in categorized.items():
            pct = len(samples) / total * 100 if total > 0 else 0
            print(f"{category:25s}: {len(samples):5d} ({pct:5.1f}%)")
        
        print(f"{'='*50}")


def sample_data_command(
    input_path: str,
    output_dir: str,
    total_samples: int = 1000,
    grpo_ratio: float = 0.5,
):
    """
    命令行入口：采样训练数据
    
    Usage:
        python -m data_processing.difficulty_sampler \
            --input data/verified_samples.jsonl \
            --output_dir data/training_sets \
            --total_samples 1000
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Sample training data by difficulty")
    parser.add_argument("--input", required=True, help="Input verified data file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--total_samples", type=int, default=1000)
    parser.add_argument("--grpo_ratio", type=float, default=0.5)
    parser.add_argument("--min_quality", type=float, default=0.6)
    args = parser.parse_args()
    
    # 加载数据
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples from {args.input}")
    
    # 设置配置
    config = SamplingConfig(
        total_samples=args.total_samples,
        grpo_ratio=args.grpo_ratio,
        min_quality=args.min_quality,
    )
    
    # 采样
    sampler = DifficultyAwareSampler(config=config)
    categorized = sampler.categorize_samples(samples)
    training_sets = sampler.sample_training_sets(categorized)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    for set_name, set_samples in training_sets.items():
        output_path = os.path.join(output_dir, f"{set_name}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in set_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved {len(set_samples)} samples to {output_path}")


if __name__ == "__main__":
    sample_data_command()
