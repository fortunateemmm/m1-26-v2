# qwen2.5-7b-instruct
# m1k
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m1k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/sft-1k \
    --exp_name 7b \
    --gradient_checkpointing False \
    --use_flash_attention_2 False

# gradient_checkpoint=True, 40GB per GPU with 8 GPUs
# gradient_checkpoint=False, 50GB per GPU with 8 GPUs

# m23k
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m23k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/sft-23k \
    --exp_name 7b \
    --gradient_checkpointing False \
    --use_flash_attention_2 False


# qwen2.5-32b-instruct
# 1k
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m1k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/sft-1k \
    --exp_name 32b \
    --gradient_checkpointing True \
    --use_flash_attention_2 False \
    --model_name "Qwen/Qwen2.5-32B-Instruct" \
    --nnodes 2 \
    --head_node_ip ???

# gradient_checkpointing=True, no cpu offload, 50-60GB per GPU with 16 GPUs


# =============================================================================
# Improved Pipeline: Data Preprocessing + Hybrid Training
# =============================================================================
# 
# Step 1: Preprocess raw data (quality verification + difficulty sampling)
# Step 2: Train with hybrid SFT-GRPO
#
# Usage:
#   1. First generate/reasoning data using existing pipeline
#   2. Run preprocessing to filter and sample
#   3. Run improved training

# -----------------------------------------------------------------------------
# Step 1: Data Preprocessing
# -----------------------------------------------------------------------------

# Preprocess raw data (example with m1k)
# 数据预处理：质量验证 + 难度计算
# 输出统一数据集，训练时会根据难度动态选择SFT或GRPO
python src/data_processing/process_data.py \
    --input data/raw_reasoning_samples.jsonl \
    --output data/processed/m1k_processed.jsonl \
    --min_total 0.55 \
    --min_difficulty 0.3

# This creates:
#   - data/processed/m1k_processed.jsonl (统一数据集，包含difficulty字段)

# -----------------------------------------------------------------------------
# Step 2: Improved Training with Pre-processed Data
# -----------------------------------------------------------------------------
# 
# 使用统一数据集，训练时根据per-sample loss动态选择SFT或GRPO：
# - loss < threshold (简单样本) → SFT
# - loss >= threshold (困难样本) → GRPO

# 7B with improved pipeline (using unified pre-processed data)
bash src/train/sft_local.sh \
    --gpu_count 8 \
    --output_dir outputs/improved-1k \
    --exp_name 7b-improved \
    --gradient_checkpointing False \
    --use_flash_attention_2 False \
    --use_hybrid_training True \
    --loss_threshold 2.0 \
    --grpo_num_generations 8 \
    --grpo_temperature 0.7 \
    --grpo_learning_rate 1e-6 \
    --reward_answer_weight 1.0 \
    --reward_format_weight 0.2 \
    --reward_semantic_weight 0.3 \
    --train_data_path data/processed/m1k_processed.jsonl

# =============================================================================
# Advanced Training: Curriculum + Adaptive Threshold + Hard Sample Replay
# =============================================================================
# 
# 高级特性：
# - 课程学习：从简单到困难逐步训练
# - 自适应Loss阈值：初期高阈值，后期低阈值
# - 困难样本回放：保留困难样本反复训练
# - GRPO温度退火：后期降低探索，促进收敛

# 7B with advanced training
bash src/train/sft_local.sh \
    --gpu_count 8 \
    --output_dir outputs/advanced-1k \
    --exp_name 7b-advanced \
    --gradient_checkpointing False \
    --use_flash_attention_2 False \
    --use_hybrid_training True \
    --use_advanced_training True \
    --use_curriculum True \
    --use_hard_sample_replay True \
    --use_adaptive_threshold True \
    --initial_loss_threshold 3.0 \
    --final_loss_threshold 1.5 \
    --replay_buffer_size 500 \
    --grpo_num_generations 8 \
    --grpo_temperature 0.7 \
    --grpo_learning_rate 1e-6 \
    --reward_answer_weight 1.0 \
    --reward_format_weight 0.2 \
    --reward_semantic_weight 0.3 \
    --train_data_path data/processed/m1k_processed.jsonl \
    --epochs 5


# =============================================================================
# Hybrid SFT-GRPO Training
# =============================================================================

# m1k with hybrid training (7B)
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m1k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/hybrid-1k \
    --exp_name 7b-hybrid \
    --gradient_checkpointing False \
    --use_flash_attention_2 False \
    --use_hybrid_training True \
    --loss_threshold 2.0 \
    --grpo_num_generations 8 \
    --grpo_temperature 0.7 \
    --grpo_learning_rate 1e-6 \
    --reward_answer_weight 1.0 \
    --reward_format_weight 0.2 \
    --reward_semantic_weight 0.3

# m23k with hybrid training (7B)
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m23k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/hybrid-23k \
    --exp_name 7b-hybrid \
    --gradient_checkpointing False \
    --use_flash_attention_2 False \
    --use_hybrid_training True \
    --loss_threshold 2.0 \
    --grpo_num_generations 8 \
    --grpo_temperature 0.7 \
    --grpo_learning_rate 1e-6 \
    --reward_answer_weight 1.0 \
    --reward_format_weight 0.2 \
    --reward_semantic_weight 0.5

# 32B with hybrid training
bash src/train/sft_local.sh \
    --train_dataset_name UCSC-VLAA/m1k-tokenized \
    --gpu_count 8 \
    --output_dir outputs/hybrid-1k \
    --exp_name 32b-hybrid \
    --gradient_checkpointing True \
    --use_flash_attention_2 False \
    --model_name "Qwen/Qwen2.5-32B-Instruct" \
    --nnodes 2 \
    --head_node_ip ??? \
    --use_hybrid_training True \
    --loss_threshold 2.0 \
    --grpo_num_generations 4 \
    --grpo_temperature 0.7 \
    --grpo_learning_rate 5e-7 \
    --reward_answer_weight 1.0 \
    --reward_format_weight 0.2 \
    --reward_semantic_weight 0.3
