"""
数据预处理主脚本

整合质量验证和难度感知采样，生成训练所需的数据集

流程:
1. 加载原始数据
2. 质量验证
3. 难度计算和采样
4. 输出GRPO训练集和SFT训练集
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataProcessingConfig:
    """数据处理配置"""
    # 输入
    input_path: str = "data/raw_samples.jsonl"
    
    # 质量验证参数
    min_format_score: float = 0.6
    min_coherence_score: float = 0.5
    min_consistency_score: float = 0.6
    min_medical_score: float = 0.3
    min_total_score: float = 0.55
    
    # 采样参数
    total_samples: int = 1000
    grpo_ratio: float = 0.5
    difficulty_bins: int = 5
    balance_across_difficulty: bool = True
    
    # 输出
    output_dir: str = "data/processed"


def process_data(config: DataProcessingConfig) -> Dict[str, str]:
    """
    执行完整的数据处理流程
    
    Returns:
        各数据集的输出路径
    """
    from .quality_verifier import ReasoningQualityVerifier, QualityThresholds
    from .difficulty_sampler import DifficultyAwareSampler, SamplingConfig
    
    print(f"\n{'='*60}")
    print(f"Data Processing Pipeline")
    print(f"{'='*60}")
    
    # Step 1: 加载数据
    print(f"\n[Step 1] Loading data from {config.input_path}...")
    samples = []
    with open(config.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"    Loaded {len(samples)} samples")
    
    # Step 2: 质量验证
    print(f"\n[Step 2] Verifying reasoning quality...")
    thresholds = QualityThresholds(
        min_format_score=config.min_format_score,
        min_coherence_score=config.min_coherence_score,
        min_consistency_score=config.min_consistency_score,
        min_medical_score=config.min_medical_score,
        min_total_score=config.min_total_score,
    )
    
    verifier = ReasoningQualityVerifier(thresholds)
    verified_samples, verification_stats = verifier.verify_dataset(samples)
    
    print(f"    Total: {verification_stats['total']}")
    print(f"    Valid: {verification_stats['valid']} ({verification_stats['pass_rate']*100:.1f}%)")
    print(f"    Invalid: {verification_stats['invalid']}")
    print(f"    Average score: {verification_stats['avg_score']:.3f}")
    
    # 只保留通过验证的样本
    valid_samples = [s for s in verified_samples if s["is_valid"]]
    
    if len(valid_samples) < config.total_samples:
        logger.warning(
            f"Only {len(valid_samples)} valid samples, but need {config.total_samples}. "
            f"Consider lowering quality thresholds."
        )
    
    # Step 3: 难度感知采样
    print(f"\n[Step 3] Difficulty-aware sampling...")
    sampling_config = SamplingConfig(
        total_samples=min(config.total_samples, len(valid_samples)),
        grpo_ratio=config.grpo_ratio,
        difficulty_bins=config.difficulty_bins,
        min_quality=config.min_total_score,
        balance_across_difficulty=config.balance_across_difficulty,
    )
    
    sampler = DifficultyAwareSampler(config=sampling_config)
    categorized = sampler.categorize_samples(valid_samples)
    training_sets = sampler.sample_training_sets(categorized)
    
    # Step 4: 保存结果
    print(f"\n[Step 4] Saving results to {config.output_dir}...")
    os.makedirs(config.output_dir, exist_ok=True)
    
    output_paths = {}
    for set_name, set_samples in training_sets.items():
        output_path = os.path.join(config.output_dir, f"{set_name}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in set_samples:
                # 清理输出字段
                clean_sample = {
                    "question": sample.get("question", ""),
                    "prompt": sample.get("prompt", sample.get("question", "")),
                    "reasoning": sample.get("reasoning", ""),
                    "answer": sample.get("answer", sample.get("distilled_answer_string", "")),
                    "difficulty": sample.get("difficulty", 0),
                    "quality": sample.get("quality", 0),
                }
                f.write(json.dumps(clean_sample, ensure_ascii=False) + '\n')
        
        output_paths[set_name] = output_path
        print(f"    {set_name}: {len(set_samples)} samples -> {output_path}")
    
    # 保存验证统计
    stats_path = os.path.join(config.output_dir, "processing_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            "verification": verification_stats,
            "sampling": {
                "total_requested": config.total_samples,
                "grpo_set_size": len(training_sets["grpo_set"]),
                "sft_set_size": len(training_sets["sft_set"]),
                "rejection_set_size": len(training_sets["rejection_set"]),
            }
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Data processing complete!")
    print(f"{'='*60}\n")
    
    return output_paths


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process training data")
    
    # 输入输出
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    
    # 质量参数
    parser.add_argument("--min_format", type=float, default=0.6)
    parser.add_argument("--min_coherence", type=float, default=0.5)
    parser.add_argument("--min_consistency", type=float, default=0.6)
    parser.add_argument("--min_medical", type=float, default=0.3)
    parser.add_argument("--min_total", type=float, default=0.55)
    
    # 采样参数
    parser.add_argument("--total_samples", type=int, default=1000)
    parser.add_argument("--grpo_ratio", type=float, default=0.5)
    parser.add_argument("--difficulty_bins", type=int, default=5)
    
    args = parser.parse_args()
    
    config = DataProcessingConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        min_format_score=args.min_format,
        min_coherence_score=args.min_coherence,
        min_consistency_score=args.min_consistency,
        min_medical_score=args.min_medical,
        min_total_score=args.min_total,
        total_samples=args.total_samples,
        grpo_ratio=args.grpo_ratio,
        difficulty_bins=args.difficulty_bins,
    )
    
    process_data(config)


if __name__ == "__main__":
    main()
