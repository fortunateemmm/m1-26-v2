# 改进方案：基于流程图的系统性优化

## 1. 改进后的完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           改进后的Data Curation流程                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐                 │
│  │ 196K Medical │────▶│    Difficulty    │────▶│      Thinking    │                 │
│  │   QA Pairs   │     │    Filtering     │     │    Generation    │                 │
│  └──────────────┘     │    (37K)         │     │    (DeepSeek-R1) │                 │
│                       └──────────────────┘     └────────┬─────────┘                 │
│                                                         │                           │
│                                                         ▼                           │
│                               ┌─────────────────────────────────────────────┐       │
│                               │   [新] Reasoning Quality Verification       │       │
│                               │   • Rule-based format check                 │       │
│                               │   • LLM-as-Judge semantic check             │       │
│                               │   • Self-consistency verification           │       │
│                               └─────────────────────┬───────────────────────┘       │
│                                                     │                               │
│                              ┌──────────────────────┼──────────────────────┐        │
│                              ▼                      ▼                      ▼        │
│                    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│
│                    │ High Quality    │    │  Medium Quality │    │   Low Quality   ││
│                    │ Reasoning       │    │  Reasoning      │    │   Reasoning     ││
│                    │ (23K)           │    │  (10K)          │    │   (Drop)        ││
│                    └────────┬────────┘    └────────┬────────┘    └─────────────────┘│
│                             │                      │                                  │
│                             ▼                      ▼                                  │
│                    ┌─────────────────────────────────────────────────────┐            │
│                    │   [新] Difficulty-Aware Quality Sampling            │            │
│                    │   • High Quality + Hard → GRPO Training Set         │            │
│                    │   • High Quality + Easy → SFT Training Set          │            │
│                    │   • Medium Quality → Augmentation Pool              │            │
│                    └──────────────────────┬──────────────────────────────┘            │
│                                           │                                           │
│                      ┌────────────────────┼────────────────────┐                      │
│                      ▼                    ▼                    ▼                      │
│               ┌────────────┐       ┌────────────┐       ┌────────────┐               │
│               │ 1K Hard    │       │ 1K Easy    │       │ 23K Mixed  │               │
│               │ GRPO Set   │       │ SFT Set    │       │ SFT Set    │               │
│               └────────────┘       └────────────┘       └────────────┘               │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           改进后的Model Training流程                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   Phase 1: Warm-up SFT                   Phase 2: Hybrid SFT-GRPO                   │
│   ┌─────────────────────────┐            ┌─────────────────────────┐                │
│   │ Easy Samples (1K)       │            │ All Samples             │                │
│   │         ↓               │            │         ↓               │                │
│   │     SFT Training        │ ─────────▶ │  Per-sample Loss Check  │                │
│   │     (3 epochs)          │            │         ↓               │                │
│   └─────────────────────────┘            │  ┌─────────────────┐    │                │
│                                          │  │ loss < τ        │    │                │
│                                          │  │ → SFT Update    │    │                │
│                                          │  ├─────────────────┤    │                │
│                                          │  │ loss ≥ τ        │    │                │
│                                          │  │ → GRPO Update   │    │                │
│                                          │  └─────────────────┘    │                │
│                                          └─────────────────────────┘                │
│                                                                                      │
│   Phase 3: Self-Verification Fine-tuning [新]                                        │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  Generate → Verify → Reward → Fine-tune                                    │   │
│   │  (模型学会生成后自我验证reasoning的正确性)                                     │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 五个核心改进点

### 改进1：Reasoning Quality Verification（推理质量验证）

**问题**：原始流程只看答案正确性，不看reasoning质量

**解决方案**：在Thinking Generation后增加质量验证层

```python
class ReasoningQualityVerifier:
    """
    验证生成的reasoning质量
    """
    def verify(self, reasoning: str, question: str, answer: str) -> Dict[str, float]:
        return {
            "format_score": self.check_format(reasoning),      # 格式完整性
            "coherence_score": self.check_coherence(reasoning), # 逻辑连贯性
            "relevance_score": self.check_relevance(reasoning, question), # 相关性
            "consistency_score": self.check_consistency(reasoning, answer), # 与答案一致性
            "medical_accuracy": self.check_medical_terms(reasoning), # 医学术语准确性
        }
    
    def check_format(self, reasoning: str) -> float:
        """检查推理格式是否完整"""
        score = 0.0
        # 检查是否有清晰的步骤
        if re.search(r'step\s*\d|first|second|third|finally', reasoning.lower()):
            score += 0.25
        # 检查是否有因果连接词
        if re.search(r'therefore|because|thus|hence|consequently', reasoning.lower()):
            score += 0.25
        # 检查长度合理性
        word_count = len(reasoning.split())
        if 50 < word_count < 1000:
            score += 0.25
        # 检查是否有医学术语
        medical_terms = ['diagnosis', 'treatment', 'symptom', 'pathology', ...]
        if any(term in reasoning.lower() for term in medical_terms):
            score += 0.25
        return score
    
    def check_consistency(self, reasoning: str, answer: str) -> float:
        """检查推理与答案的一致性"""
        # 使用LLM验证推理是否支持答案
        prompt = f"""
        Reasoning: {reasoning}
        Final Answer: {answer}
        
        Does the reasoning logically lead to the answer? 
        Rate 0-1 for consistency.
        """
        return self.llm_judge(prompt)
```

### 改进2：Curriculum Learning（课程学习）

**问题**：原始流程没有考虑训练顺序

**解决方案**：先简单后困难，渐进式训练

```python
class CurriculumScheduler:
    """
    课程学习调度器：先易后难
    """
    def __init__(self, dataset, difficulty_scores):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.current_stage = 0
        self.stages = [
            {"difficulty_range": (0, 0.3), "epochs": 2},   # Stage 0: 简单样本
            {"difficulty_range": (0.3, 0.6), "epochs": 2}, # Stage 1: 中等样本
            {"difficulty_range": (0.6, 1.0), "epochs": 3}, # Stage 2: 困难样本
        ]
    
    def get_stage_data(self, stage: int):
        """获取当前阶段的数据"""
        config = self.stages[stage]
        low, high = config["difficulty_range"]
        mask = (self.difficulty_scores >= low) & (self.difficulty_scores < high)
        return self.dataset[mask], config["epochs"]
    
    def should_advance(self, metrics: Dict) -> bool:
        """判断是否应该进入下一阶段"""
        # 当前阶段的loss已经收敛
        return metrics["loss"] < self.threshold
```

### 改进3：Difficulty-Aware Sampling（难度感知采样）

**问题**：原始Diversity Sampling不考虑reasoning难度

**解决方案**：基于难度和质量的分层采样

```python
class DifficultyAwareSampler:
    """
    难度感知的分层采样器
    """
    def __init__(self, difficulty_bins=5, quality_threshold=0.7):
        self.difficulty_bins = difficulty_bins
        self.quality_threshold = quality_threshold
    
    def sample(self, dataset, n_samples=1000):
        """
        分层采样策略：
        - 高质量 + 高难度 → GRPO训练集
        - 高质量 + 低难度 → SFT训练集
        - 中等质量 → 数据增强池
        """
        # 计算每个样本的difficulty score和quality score
        difficulties = self.compute_difficulty(dataset)
        qualities = self.compute_quality(dataset)
        
        # 分层采样
        hard_high_quality = self.filter_and_sample(
            dataset, difficulties > 0.7, qualities > self.quality_threshold,
            n_samples=n_samples // 2
        )
        
        easy_high_quality = self.filter_and_sample(
            dataset, difficulties <= 0.7, qualities > self.quality_threshold,
            n_samples=n_samples // 2
        )
        
        return {
            "grpo_set": hard_high_quality,
            "sft_set": easy_high_quality,
        }
    
    def compute_difficulty(self, sample):
        """计算样本难度"""
        # 基于多个因素：
        # 1. 原始模型是否能解决
        # 2. 生成reasoning的长度（越长可能越难）
        # 3. 问题类型（多选题 vs 简答题）
        pass
    
    def compute_quality(self, sample):
        """计算reasoning质量"""
        # 基于ReasoningQualityVerifier的分数
        pass
```

### 改进4：Self-Verification Training（自我验证训练）

**问题**：模型不会自我验证生成的reasoning是否正确

**解决方案**：训练模型学会自我验证

```python
class SelfVerificationTrainer:
    """
    自我验证训练：让模型学会检查自己的推理
    """
    def create_verification_data(self, dataset):
        """
        创建验证数据格式：
        Input: [问题 + 生成的推理]
        Output: [验证结果 + 修正（如有）]
        """
        verification_examples = []
        for sample in dataset:
            question = sample["question"]
            reasoning = sample["reasoning"]
            correct_answer = sample["answer"]
            
            # 生成验证样本
            verification_examples.append({
                "input": f"Question: {question}\nReasoning: {reasoning}\n\nIs this reasoning correct? Explain why.",
                "output": self.generate_verification(reasoning, correct_answer)
            })
        
        return verification_examples
    
    def generate_verification(self, reasoning, correct_answer):
        """生成验证结果"""
        # 检查推理中的关键步骤是否正确
        # 指出可能的错误
        # 提供修正建议
        pass

class SelfVerificationGRPO(GRPOTrainer):
    """
    带自我验证的GRPO训练器
    """
    def compute_rewards(self, prompt, response, ground_truth):
        """
        扩展的reward计算，包含自我验证
        """
        base_rewards = super().compute_rewards(prompt, response, ground_truth)
        
        # 自我验证奖励
        verification_prompt = f"""
        Question: {prompt}
        My reasoning: {response}
        
        Please verify: Is my reasoning correct? 
        If incorrect, explain the error.
        """
        
        verification = self.generate(verification_prompt)
        verification_score = self.evaluate_verification(verification)
        
        # 如果模型能正确识别自己的错误，给予额外奖励
        if self.can_self_correct(verification, ground_truth):
            base_rewards["self_verification"] = 1.0
        else:
            base_rewards["self_verification"] = 0.0
        
        return base_rewards
```

### 改进5：Multi-Teacher Distillation（多教师蒸馏）

**问题**：原始流程只用DeepSeek-R1生成reasoning，存在bias

**解决方案**：使用多个教师模型，集成不同视角

```python
class MultiTeacherDistiller:
    """
    多教师蒸馏：集成多个模型的reasoning
    """
    def __init__(self, teachers: List[str]):
        self.teachers = teachers  # ['deepseek-r1', 'qwen-32b', 'gpt-4', 'claude-3']
    
    def distill(self, question: str, n_samples_per_teacher: int = 3):
        """
        从多个教师模型生成reasoning
        """
        all_reasonings = []
        
        for teacher in self.teachers:
            reasonings = self.generate_from_teacher(
                teacher, question, n_samples=n_samples_per_teacher
            )
            all_reasonings.extend(reasonings)
        
        # 选择最佳reasoning（多数投票 + 质量评估）
        best_reasoning = self.select_best(all_reasonings)
        
        # 如果有分歧，生成consensus reasoning
        if self.has_disagreement(all_reasonings):
            consensus = self.generate_consensus(all_reasonings)
            return consensus
        
        return best_reasoning
    
    def select_best(self, reasonings):
        """选择最佳reasoning"""
        # 1. 多数投票选择答案
        answers = [self.extract_answer(r) for r in reasonings]
        majority_answer = max(set(answers), key=answers.count)
        
        # 2. 选择支持该答案且质量最高的reasoning
        supporting = [r for r, a in zip(reasonings, answers) if a == majority_answer]
        return max(supporting, key=lambda r: self.quality_score(r))
    
    def generate_consensus(self, reasonings):
        """生成共识reasoning"""
        prompt = f"""
        Multiple experts have analyzed this medical question with different reasonings:
        
        {self.format_reasonings(reasonings)}
        
        Please synthesize a consensus reasoning that incorporates the best elements from each.
        """
        return self.generate(prompt)
```

## 3. 改进后的完整训练配置

```yaml
# configs/improved_training.yaml

data_curation:
  difficulty_filtering:
    min_difficulty: 0.3  # 降低门槛，保留更多中等难度样本
    use_multiple_models: true  # 使用多个模型判断难度
    
  thinking_generation:
    teachers: ["deepseek-r1", "qwen-32b"]  # 多教师
    samples_per_teacher: 3
    temperature: 0.7
    
  quality_verification:
    enabled: true
    min_format_score: 0.6
    min_coherence_score: 0.5
    min_consistency_score: 0.7
    use_llm_judge: true
    
  diversity_sampling:
    strategy: "difficulty_aware"  # 难度感知采样
    difficulty_bins: 5
    balance_across_bins: true

training:
  phase_1_warmup:
    enabled: true
    data: "easy_samples"
    epochs: 2
    learning_rate: 1e-5
    
  phase_2_hybrid:
    enabled: true
    data: "all_samples"
    loss_threshold: 2.0
    grpo_config:
      num_generations: 8
      temperature: 0.7
      reward_weights:
        answer: 1.0
        format: 0.2
        semantic: 0.3
        self_verification: 0.2  # 新增
    
  phase_3_verification:
    enabled: true
    data: "verification_examples"
    epochs: 1
    learning_rate: 5e-6
```

## 4. 改进效果预期

| 指标 | 原始方法 | 改进方法 |
|------|----------|----------|
| Reasoning正确率 | ~70% | ~85% |
| 格式合规率 | ~80% | ~95% |
| 语义连贯性 | 中等 | 高 |
| 困难样本提升 | 有限 | 显著 |
| 模型自我验证能力 | 无 | 有 |
