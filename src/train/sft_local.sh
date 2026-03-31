#!/bin/bash

# Set training parameters
lr=1e-5
epochs=5
global_batch_size=16
per_device_batch_size=1
weight_decay=1e-4
train_dataset_name="mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-tokenized-120325"
uid="$(date +%Y%m%d_%H%M%S)"
model_name="Qwen/Qwen2.5-7B-Instruct"
nnodes=1
head_node_ip=localhost
gpu_count=8
output_dir="outputs/"
exp_name="sft"
gradient_checkpointing=False
use_flash_attention_2=False
port=29500

# Hybrid training parameters
use_hybrid_training=False
loss_threshold=2.0
grpo_num_generations=8
grpo_temperature=0.7
grpo_learning_rate=1e-6
grpo_clip_epsilon=0.2
grpo_kl_coeff=0.01
reward_answer_weight=1.0
reward_format_weight=0.2
reward_semantic_weight=0.3

# Pre-processed data path (for improved training)
# 使用统一数据集，训练时根据难度动态选择SFT或GRPO
train_data_path=""

# Advanced training features
use_advanced_training=False
use_curriculum=True
use_hard_sample_replay=True
use_adaptive_threshold=True
initial_loss_threshold=3.0
final_loss_threshold=1.5
replay_buffer_size=500

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --global_batch_size) global_batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --train_dataset_name) train_dataset_name="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        --per_device_batch_size) per_device_batch_size="$2"; shift 2 ;;
        --gpu_count) gpu_count="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --exp_name) exp_name="$2"; shift 2 ;;
        --nnodes) nnodes="$2"; shift 2 ;;
        --head_node_ip) head_node_ip="$2"; shift 2 ;;
        --gradient_checkpointing) gradient_checkpointing="$2"; shift 2 ;;
        --use_flash_attention_2) use_flash_attention_2="$2"; shift 2 ;;
        --model_name) model_name="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --use_hybrid_training) use_hybrid_training="$2"; shift 2 ;;
        --loss_threshold) loss_threshold="$2"; shift 2 ;;
        --grpo_num_generations) grpo_num_generations="$2"; shift 2 ;;
        --grpo_temperature) grpo_temperature="$2"; shift 2 ;;
        --grpo_learning_rate) grpo_learning_rate="$2"; shift 2 ;;
        --grpo_clip_epsilon) grpo_clip_epsilon="$2"; shift 2 ;;
        --grpo_kl_coeff) grpo_kl_coeff="$2"; shift 2 ;;
        --reward_answer_weight) reward_answer_weight="$2"; shift 2 ;;
        --reward_format_weight) reward_format_weight="$2"; shift 2 ;;
        --reward_semantic_weight) reward_semantic_weight="$2"; shift 2 ;;
        --train_data_path) train_data_path="$2"; shift 2 ;;
        --use_advanced_training) use_advanced_training="$2"; shift 2 ;;
        --use_curriculum) use_curriculum="$2"; shift 2 ;;
        --use_hard_sample_replay) use_hard_sample_replay="$2"; shift 2 ;;
        --use_adaptive_threshold) use_adaptive_threshold="$2"; shift 2 ;;
        --initial_loss_threshold) initial_loss_threshold="$2"; shift 2 ;;
        --final_loss_threshold) final_loss_threshold="$2"; shift 2 ;;
        --replay_buffer_size) replay_buffer_size="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Get node information
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600


# Calculate gradient accumulation steps
grad_acc=$((global_batch_size/(gpu_count * nnodes)))

echo "Number of nodes: $nnodes"
echo "Number of GPUs per node: $gpu_count"
echo "Head node IP: $head_node_ip"

# Launch distributed training using srun
if [ "$use_hybrid_training" = "True" ]; then
    run_name="hybrid_${train_dataset_name}_bs${global_batch_size}_lr${lr}_epoch${epochs}_wd${weight_decay}_thresh${loss_threshold}_${uid}"
else
    run_name="qwen_${train_dataset_name}_bs${global_batch_size}_lr${lr}_epoch${epochs}_wd${weight_decay}_${uid}"
fi

# NOTE: if we start the job with srun, no srun in the script is needed. If we start the job with sbatch, we need to use srun.
torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpu_count \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:$port \
    src/train/sft.py \
    --per_device_train_batch_size=${per_device_batch_size} \
    --per_device_eval_batch_size=${per_device_batch_size} \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=${epochs} \
    --train_file_path="${train_dataset_name}" \
    --model_name=$model_name \
    --warmup_ratio=0.05 \
    --report_to="none" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="src/train/fsdp_config_qwen.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="epoch" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="${output_dir}/${exp_name}/${run_name}" \
    --push_to_hub=false \
    --save_only_model=True \
    --gradient_checkpointing=${gradient_checkpointing} \
    --report_to='wandb' \
    --use_flash_attention_2=${use_flash_attention_2} \
    --use_hybrid_training=${use_hybrid_training} \
    --loss_threshold=${loss_threshold} \
    --grpo_num_generations=${grpo_num_generations} \
    --grpo_temperature=${grpo_temperature} \
    --grpo_learning_rate=${grpo_learning_rate} \
    --grpo_clip_epsilon=${grpo_clip_epsilon} \
    --grpo_kl_coeff=${grpo_kl_coeff} \
    --reward_answer_weight=${reward_answer_weight} \
    --reward_format_weight=${reward_format_weight} \
    --reward_semantic_weight=${reward_semantic_weight} \
    --train_data_path=${train_data_path} \
    --use_advanced_training=${use_advanced_training} \
    --use_curriculum=${use_curriculum} \
    --use_hard_sample_replay=${use_hard_sample_replay} \
    --use_adaptive_threshold=${use_adaptive_threshold} \
    --initial_loss_threshold=${initial_loss_threshold} \
    --final_loss_threshold=${final_loss_threshold} \
    --replay_buffer_size=${replay_buffer_size} \

    # --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}' \
