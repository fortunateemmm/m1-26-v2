import dotenv


dotenv.load_dotenv()


import os
from copy import deepcopy


print(os.getenv("HF_HOME"))
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional, List


warnings.filterwarnings("ignore", category=FutureWarning)
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
import trl
from datasets import DatasetDict, concatenate_datasets, load_dataset
import torch
from torch.utils.data import DataLoader

import transformers

# Import hybrid training components
from grpo_trainer import HybridSFTGRPOTrainer, GRPOConfig
from reward_functions import RewardComputer


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    train_file_path: Optional[str] = field(default="mmqm/m194k_r1_tokenized-250224")
    dagger: bool = field(default=False)
    use_flash_attention_2: bool = field(default=False)
    
    # Hybrid training parameters
    use_hybrid_training: bool = field(default=False)
    loss_threshold: float = field(default=2.0)
    grpo_num_generations: int = field(default=8)
    grpo_temperature: float = field(default=0.7)
    grpo_learning_rate: float = field(default=1e-6)
    grpo_clip_epsilon: float = field(default=0.2)
    grpo_kl_coeff: float = field(default=0.01)
    reward_answer_weight: float = field(default=1.0)
    reward_format_weight: float = field(default=0.2)
    reward_semantic_weight: float = field(default=0.3)
    
    # Pre-processed data path (for improved training)
    # 使用统一数据集，训练时根据难度动态选择SFT或GRPO
    train_data_path: Optional[str] = field(default=None)
    
    # Advanced training features
    use_advanced_training: bool = field(default=False)  # 使用高级训练器
    use_curriculum: bool = field(default=True)  # 课程学习
    use_hard_sample_replay: bool = field(default=True)  # 困难样本回放
    use_adaptive_threshold: bool = field(default=True)  # 自适应loss阈值
    initial_loss_threshold: float = field(default=3.0)  # 初始loss阈值
    final_loss_threshold: float = field(default=1.5)  # 最终loss阈值
    replay_buffer_size: int = field(default=500)  # 回放缓冲区大小


def extract_prompt_from_text(text: str, response_template: str) -> str:
    """Extract user prompt from formatted text."""
    if response_template in text:
        return text.split(response_template)[0].strip()
    return text


def extract_ground_truth_from_text(text: str, response_template: str) -> str:
    """Extract ground truth answer from formatted text."""
    if response_template in text:
        response = text.split(response_template)[1]
        # Try to extract answer from \boxed{}
        import re
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()
    return ""


def run_hybrid_training(
    model,
    tokenizer,
    dataset,
    config: TrainingConfig,
    args,
    collator,
    instruction_template: str,
    response_template: str,
):
    """Run hybrid SFT-GRPO training."""
    logging.info("Starting hybrid SFT-GRPO training...")
    
    # Create reference model for KL penalty
    logging.info("Creating reference model...")
    ref_model = deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Create optimizers
    sft_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    grpo_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.grpo_learning_rate,
    )
    
    # Create GRPO config
    grpo_config = GRPOConfig(
        num_generations=config.grpo_num_generations,
        temperature=config.grpo_temperature,
        clip_epsilon=config.grpo_clip_epsilon,
        kl_coeff=config.grpo_kl_coeff,
        loss_threshold=config.loss_threshold,
        reward_weights={
            "answer": config.reward_answer_weight,
            "format": config.reward_format_weight,
            "semantic": config.reward_semantic_weight,
        }
    )
    
    # Create hybrid trainer
    hybrid_trainer = HybridSFTGRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        grpo_config=grpo_config,
        sft_optimizer=sft_optimizer,
        grpo_optimizer=grpo_optimizer,
    )
    
    # Prepare dataloader
    train_dataset = dataset["train"]
    
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.block_size,
            return_tensors="pt",
        )
        
        # Create labels (mask prompt tokens)
        labels = encodings["input_ids"].clone()
        
        # Find response start positions
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        
        for i, text in enumerate(texts):
            # Find where response starts
            input_ids = encodings["input_ids"][i]
            response_start = -1
            
            for j in range(len(input_ids) - len(response_template_ids)):
                if input_ids[j:j+len(response_template_ids)].tolist() == response_template_ids:
                    response_start = j + len(response_template_ids)
                    break
            
            if response_start > 0:
                labels[i, :response_start] = -100
            
            # Mask padding tokens
            pad_mask = encodings["attention_mask"][i] == 0
            labels[i][pad_mask] = -100
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
            "texts": texts,
        }
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(int(args.num_train_epochs)):
        logging.info(f"Epoch {epoch + 1}/{int(args.num_train_epochs)}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move to device
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]
            
            # Extract prompts, questions, and ground truths
            prompts = []
            questions = []
            ground_truths = []
            
            for text in texts:
                prompt = extract_prompt_from_text(text, response_template)
                gt = extract_ground_truth_from_text(text, response_template)
                prompts.append(prompt)
                questions.append(prompt)
                ground_truths.append(gt)
            
            # Hybrid step
            stats = hybrid_trainer.hybrid_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prompts=prompts,
                ground_truths=ground_truths,
                questions=questions,
            )
            
            global_step += 1
            
            # Logging
            if global_step % args.logging_steps == 0:
                log_str = f"Step {global_step} | "
                log_str += " | ".join([f"{k}: {v:.4f}" for k, v in stats.items()])
                logging.info(log_str)
        
        # Save checkpoint at end of each epoch
        if args.save_strategy == "epoch":
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logging.info(f"Saved checkpoint to {checkpoint_dir}")
    
    return hybrid_trainer


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if config.use_flash_attention_2:
        logging.info(f"Use flash_attention_2")
        kwargs["attn_implementation"] =  "flash_attention_2"
    else:
        logging.info(f"Disable flash_attention_2")

    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs.update({
            "device_map": "auto",
            "torch_dtype": "auto",
            "use_cache": False,
        })
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        # NOTE xk: In s1, flash-attn is not used.
        # kwargs = {"torch_dtype": "auto", "attn_implementation": "flash_attention_2", "use_cache": False}
        kwargs = {}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size
    
    if config.use_hybrid_training:
        # Use hybrid SFT-GRPO training
        if config.train_data_path:
            # 使用预处理后的统一数据集
            logging.info(f"Using pre-processed data: {config.train_data_path}")
            
            if config.use_advanced_training:
                # 使用高级训练器（课程学习+自适应阈值+回放）
                logging.info(f"Training strategy: Advanced (curriculum + adaptive threshold + replay)")
                
                from advanced_trainer import AdvancedHybridTrainer, AdvancedTrainingConfig, UnifiedDataset
                
                advanced_config = AdvancedTrainingConfig(
                    learning_rate=args.learning_rate,
                    grpo_learning_rate=config.grpo_learning_rate,
                    weight_decay=args.weight_decay,
                    total_epochs=int(args.num_epochs),
                    initial_loss_threshold=config.initial_loss_threshold,
                    final_loss_threshold=config.final_loss_threshold,
                    threshold_annealing=config.use_adaptive_threshold,
                    use_curriculum=config.use_curriculum,
                    use_hard_sample_replay=config.use_hard_sample_replay,
                    replay_buffer_size=config.replay_buffer_size,
                    grpo_num_generations=config.grpo_num_generations,
                    initial_grpo_temperature=config.grpo_temperature,
                    reward_answer_weight=config.reward_answer_weight,
                    reward_format_weight=config.reward_format_weight,
                    reward_semantic_weight=config.reward_semantic_weight,
                    data_path=config.train_data_path,
                )
                
                trainer = AdvancedHybridTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    config=advanced_config,
                )
                
                # 加载数据集
                import json
                samples = []
                with open(config.train_data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                
                train_dataset = UnifiedDataset(samples, tokenizer, max_length=config.block_size)
                
                trainer.train(
                    train_dataset=train_dataset,
                    batch_size=args.per_device_train_batch_size,
                )
            else:
                # 使用基础混合训练器
                logging.info(f"Training strategy: Basic hybrid SFT/GRPO")
                
                from improved_trainer import DynamicHybridTrainer, HybridTrainingConfig, UnifiedDataset
                
                hybrid_config = HybridTrainingConfig(
                    learning_rate=args.learning_rate,
                    grpo_learning_rate=config.grpo_learning_rate,
                    weight_decay=args.weight_decay,
                    num_epochs=int(args.num_epochs),
                    loss_threshold=config.loss_threshold,
                    grpo_num_generations=config.grpo_num_generations,
                    grpo_temperature=config.grpo_temperature,
                    reward_answer_weight=config.reward_answer_weight,
                    reward_format_weight=config.reward_format_weight,
                    reward_semantic_weight=config.reward_semantic_weight,
                    data_path=config.train_data_path,
                )
                
                ref_model = deepcopy(model)
                ref_model.eval()
                for param in ref_model.parameters():
                    param.requires_grad = False
                
                trainer = DynamicHybridTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    config=hybrid_config,
                    ref_model=ref_model,
                )
                
                train_dataset = UnifiedDataset(
                    data_path=config.train_data_path,
                    tokenizer=tokenizer,
                    max_length=config.block_size,
                )
                
                trainer.train(
                    train_dataset=train_dataset,
                    batch_size=args.per_device_train_batch_size,
                )
        else:
            # 使用原始数据集，动态拆分
            run_hybrid_training(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                config=config,
                args=args,
                collator=None,
                instruction_template=instruction_template,
                response_template=response_template,
            )
    else:
        # Use standard SFT training
        # Only compute loss over assistant responses
        # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
        # via labels being set to -100
        collator = trl.DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False
        )
        trainer = trl.SFTTrainer(
            model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
            args=args,
            data_collator=collator,
        )

        trainer.train()
        # final_ckpt_path = os.path.join(args.output_dir, "checkpoint-final")
        # trainer.save_model(output_dir=final_ckpt_path)
        # tokenizer.save_pretrained(final_ckpt_path)
        trainer.accelerator.wait_for_everyone()



if __name__ == "__main__":
    train()
