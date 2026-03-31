# 改进方案：统一数据集 + 动态SFT/GRPO混合训练

## 核心思想

使用同一个数据集，训练时根据per-sample loss动态选择SFT或GRPO：

```
统一数据集
    │
    ▼
计算per-sample loss
    │
    ├─ loss < threshold ──▶ SFT（简单样本，模型已能较好学习）
    │
    └─ loss >= threshold ──▶ GRPO（困难样本，需要更多探索）
```

## 流程

### Step 1: 数据预处理

```bash
# 质量验证 + 难度计算
python src/data_processing/process_data.py \
    --input data/raw_reasoning_samples.jsonl \
    --output data/processed/m1k_processed.jsonl \
    --min_total 0.55 \
    --min_difficulty 0.3
```

输出：统一数据集（包含difficulty字段）

### Step 2: 混合训练

```bash
bash src/train/sft_local.sh \
    --use_hybrid_training True \
    --loss_threshold 2.0 \
    --train_data_path data/processed/m1k_processed.jsonl
```

## 动态切换逻辑

```python
def _should_use_grpo(self, batch):
    """
    判断是否使用GRPO
    
    1. 基于预计算的difficulty做初步筛选
    2. 每N步计算真实loss做准确判断
    """
    difficulty = batch["difficulty"]
    
    # 高difficulty样本直接用GRPO
    if difficulty > 0.7:
        return True
    
    # 定期计算真实loss
    if self.step_count % 5 == 0:
        loss = compute_loss(batch)
        return loss >= self.config.loss_threshold
    
    return difficulty > 0.5
```

## 预期效果

| 样本类型 | 训练方式 | 预期效果 |
|---------|---------|---------|
| 简单（低loss）| SFT | 快速收敛，稳定学习 |
| 困难（高loss）| GRPO | 探索更好的reasoning路径 |

## 文件结构

```
src/
├── data_processing/
│   └── process_data.py      # 数据预处理（质量验证+难度计算）
└── train/
    ├── sft.py               # 主训练脚本
    ├── improved_trainer.py  # 改进后的混合训练器
    ├── grpo_trainer.py      # GRPO训练器
    └── reward_functions.py  # 奖励函数
```
