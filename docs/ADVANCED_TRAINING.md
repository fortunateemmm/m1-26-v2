# 高级训练策略

## 概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Advanced Hybrid Training                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────┐                                                           │
│  │ Curriculum       │  Stage 1: Easy (0.3-0.5) → 1 epoch                       │
│  │ Learning         │  Stage 2: Medium (0.5-0.7) → 2 epochs                    │
│  │                  │  Stage 3: Hard (0.7-1.0) → 2 epochs                       │
│  └──────────────────┘                                                           │
│                                                                                 │
│  ┌──────────────────┐  threshold = 3.0 → 1.5 (cosine annealing)                │
│  │ Adaptive Loss    │  初期：只对真正困难的样本用GRPO                            │
│  │ Threshold        │  后期：更多样本用GRPO精调                                  │
│  └──────────────────┘                                                           │
│                                                                                 │
│  ┌──────────────────┐  保留loss最高的500个样本                                   │
│  │ Hard Sample      │  每个batch插入20%回放样本                                  │
│  │ Replay           │  困难样本被反复训练                                        │
│  └──────────────────┘                                                           │
│                                                                                 │
│  ┌──────────────────┐  temperature = 0.9 → 0.5 (cosine annealing)              │
│  │ GRPO Temperature │  初期：高探索，发现更好的reasoning路径                     │
│  │ Annealing        │  后期：低探索，收敛到稳定策略                              │
│  └──────────────────┘                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 使用方法

```bash
# 高级训练（包含所有特性）
bash src/train/sft_local.sh \
    --use_hybrid_training True \
    --use_advanced_training True \
    --use_curriculum True \
    --use_hard_sample_replay True \
    --use_adaptive_threshold True \
    --initial_loss_threshold 3.0 \
    --final_loss_threshold 1.5 \
    --replay_buffer_size 500 \
    --train_data_path data/processed/m1k_processed.jsonl \
    --epochs 5
```

## 核心组件

### 1. 课程学习 (Curriculum Learning)

```python
curriculum_stages = [
    CurriculumStage("warmup", (0.3, 0.5), epochs=1, grpo_ratio=0.1),
    CurriculumStage("main_easy", (0.4, 0.6), epochs=1, grpo_ratio=0.2),
    CurriculumStage("main_medium", (0.5, 0.7), epochs=2, grpo_ratio=0.3),
    CurriculumStage("main_hard", (0.6, 0.8), epochs=2, grpo_ratio=0.4),
    CurriculumStage("final_hard", (0.7, 1.0), epochs=2, grpo_ratio=0.5),
]
```

**优势**：
- 先让模型学会简单样本，建立基础
- 逐步增加难度，避免模型崩溃
- 每个阶段可以调整GRPO比例

### 2. 自适应Loss阈值 (Adaptive Threshold)

```python
# 余弦退火
threshold(t) = final + 0.5 * (initial - final) * (1 + cos(π * t))

# 示例：3.0 → 1.5
# epoch 0: 3.0 (只对最高loss用GRPO)
# epoch 2: 2.25
# epoch 4: 1.5 (更多样本用GRPO)
```

**优势**：
- 初期：高阈值，只对真正困难的样本用GRPO，稳定训练
- 后期：低阈值，更多样本用GRPO，精细调整

### 3. 困难样本回放 (Hard Sample Replay)

```python
class ReplayBuffer:
    """保存loss最高的样本，反复训练"""
    
    def add(self, sample, loss):
        # 保存样本及其loss
    
    def sample(self, n):
        # 按loss加权采样，loss越高越可能被选中
```

**优势**：
- 困难样本不会被遗忘
- 多次训练有助于模型真正学会
- 限制缓冲区大小，避免重复过多

### 4. GRPO温度退火 (Temperature Annealing)

```python
# 温度退火
temperature(t) = final + 0.5 * (initial - final) * (1 + cos(π * t))

# 示例：0.9 → 0.5
# epoch 0: 0.9 (高探索)
# epoch 2: 0.7
# epoch 4: 0.5 (低探索)
```

**优势**：
- 初期：高温，探索多种reasoning路径
- 后期：低温，收敛到最优策略

## 预期效果

| 策略 | 原始 | 改进 | 提升 |
|------|------|------|------|
| 困难样本准确率 | 45% | 65% | +20% |
| 训练稳定性 | 中 | 高 | 显著提升 |
| 收敛速度 | 慢 | 快 | -30% epochs |
| Reasoning质量 | 中 | 高 | 显著提升 |

## 文件结构

```
src/train/
├── advanced_trainer.py    # 高级训练器（新）
├── improved_trainer.py    # 基础混合训练器
├── grpo_trainer.py        # GRPO训练器
├── reward_functions.py    # 奖励函数
└── sft.py                 # 主训练脚本
```
