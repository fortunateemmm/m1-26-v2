"""
改进后的混合训练器

使用统一数据集，根据per-sample loss动态选择SFT或GRPO：
- 简单样本（loss < threshold）：SFT
- 困难样本（loss >= threshold）：GRPO
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class HybridTrainingConfig:
    """混合训练配置"""
    learning_rate: float = 1e-5
    grpo_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    num_epochs: int = 3
    
    # 动态切换阈值
    loss_threshold: float = 2.0  # 高于此值使用GRPO
    
    # GRPO参数
    grpo_num_generations: int = 8
    grpo_temperature: float = 0.7
    grpo_clip_epsilon: float = 0.2
    grpo_entropy_coeff: float = 0.01
    
    # 奖励权重
    reward_answer_weight: float = 1.0
    reward_format_weight: float = 0.2
    reward_semantic_weight: float = 0.3
    
    # 数据路径
    data_path: Optional[str] = None


class UnifiedDataset(Dataset):
    """
    统一数据集
    
    SFT和GRPO使用同一个数据集，训练时根据难度动态选择
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
        
        # 打印难度分布
        difficulties = [s.get("difficulty", 0.5) for s in self.samples]
        logger.info(f"Difficulty distribution: min={min(difficulties):.2f}, max={max(difficulties):.2f}, mean={sum(difficulties)/len(difficulties):.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        prompt = sample.get("prompt", sample.get("question", ""))
        reasoning = sample.get("reasoning", "")
        answer = sample.get("answer", "")
        
        # 格式化为chat template
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n{answer}"},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        # 创建labels（只在assistant部分计算loss）
        labels = encoding["input_ids"].copy()
        
        # 找到assistant开始位置并mask prompt
        assistant_token = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        input_ids = encoding["input_ids"]
        
        for i in range(len(input_ids) - len(assistant_token)):
            if input_ids[i:i+len(assistant_token)] == assistant_token:
                for j in range(i):
                    labels[j] = -100
                break
        
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels),
            "prompt": prompt,
            "answer": answer,
            "difficulty": sample.get("difficulty", 0.5),
        }


class DynamicHybridTrainer:
    """
    动态混合训练器
    
    核心逻辑：
    1. 计算per-sample loss判断难度
    2. 简单样本（低loss）→ SFT更新
    3. 困难样本（高loss）→ GRPO更新
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: HybridTrainingConfig,
        ref_model=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.ref_model = ref_model
        
        # SFT优化器（用于简单样本）
        self.sft_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # GRPO优化器（用于困难样本）
        self.grpo_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.grpo_learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 统计
        self.step_count = 0
        self.sft_steps = 0
        self.grpo_steps = 0
    
    def train(
        self,
        train_dataset: UnifiedDataset,
        batch_size: int = 1,
    ):
        """执行训练"""
        logger.info(f"Starting hybrid training with {len(train_dataset)} samples")
        logger.info(f"Loss threshold: {self.config.loss_threshold}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_stats = {
                "sft_loss": [], "grpo_loss": [],
                "sft_count": 0, "grpo_count": 0,
                "hard_ratio": [],
            }
            
            for batch_idx, batch in enumerate(train_loader):
                # 动态判断使用SFT还是GRPO
                use_grpo = self._should_use_grpo(batch)
                
                if use_grpo:
                    stats = self._grpo_step(batch)
                    epoch_stats["grpo_loss"].append(stats.get("loss", 0))
                    epoch_stats["grpo_count"] += 1
                    step_type = "GRPO"
                else:
                    stats = self._sft_step(batch)
                    epoch_stats["sft_loss"].append(stats.get("loss", 0))
                    epoch_stats["sft_count"] += 1
                    step_type = "SFT"
                
                self.step_count += 1
                
                # 日志
                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"Step {self.step_count} | {step_type} | "
                        f"Loss: {stats.get('loss', 0):.4f} | "
                        f"Difficulty: {batch['difficulty'][0]:.2f}"
                    )
            
            # Epoch统计
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"  SFT steps: {epoch_stats['sft_count']}")
            logger.info(f"  GRPO steps: {epoch_stats['grpo_count']}")
            if epoch_stats['sft_loss']:
                logger.info(f"  Avg SFT loss: {sum(epoch_stats['sft_loss'])/len(epoch_stats['sft_loss']):.4f}")
            if epoch_stats['grpo_loss']:
                logger.info(f"  Avg GRPO loss: {sum(epoch_stats['grpo_loss'])/len(epoch_stats['grpo_loss']):.4f}")
    
    def _should_use_grpo(self, batch: Dict) -> bool:
        """
        判断是否应该使用GRPO
        
        方法1：基于数据集中的difficulty标记
        方法2：基于实时计算的per-sample loss
        """
        # 方法1：使用预计算的difficulty（快速）
        difficulty = batch["difficulty"][0] if isinstance(batch["difficulty"], list) else batch["difficulty"]
        
        # 高difficulty样本更可能使用GRPO
        # 但最终决定基于loss，这里做初步筛选
        if difficulty > 0.7:
            return True
        
        # 方法2：实时计算loss（更准确但更慢）
        # 可以选择性地对部分batch进行实时loss计算
        if self.step_count % 5 == 0:  # 每5步计算一次真实loss
            return self._compute_and_check_loss(batch)
        
        return difficulty > 0.5
    
    def _compute_and_check_loss(self, batch: Dict) -> bool:
        """计算per-sample loss并判断"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss.item()
        
        self.model.train()
        
        return loss >= self.config.loss_threshold
    
    def _sft_step(self, batch: Dict) -> Dict[str, float]:
        """SFT训练步骤"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        labels = batch["labels"].to(self.model.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        
        self.sft_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.sft_optimizer.step()
        
        return {"loss": loss.item()}
    
    def _grpo_step(self, batch: Dict) -> Dict[str, float]:
        """
        GRPO训练步骤
        
        对于困难样本，生成多个response并使用GRPO更新
        """
        self.model.train()
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
        answer = batch["answer"][0] if isinstance(batch["answer"], list) else batch["answer"]
        
        device = self.model.device
        G = self.config.grpo_num_generations
        
        # Step 1: 生成G个responses
        with torch.no_grad():
            expanded_input_ids = input_ids.repeat(G, 1).to(device)
            expanded_attention_mask = attention_mask.repeat(G, 1).to(device)
            
            outputs = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=1024,
                temperature=self.config.grpo_temperature,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        sequences = outputs.sequences
        
        # Step 2: 解析responses
        gen_tokens = sequences[:, input_ids.shape[1]:]
        responses = [self.tokenizer.decode(gen_tokens[i], skip_special_tokens=True) for i in range(G)]
        
        # Step 3: 计算rewards
        rewards = self._compute_rewards(responses, prompt, answer)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # Step 4: 计算GRPO loss
        logits = self.model(sequences).logits[:, -gen_tokens.shape[1]-1:-1, :]
        token_logprobs = F.log_softmax(logits, dim=-1)
        token_logprobs = token_logprobs.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Group-relative advantages
        rewards_grouped = rewards.view(1, G)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_grouped - group_mean) / group_std).view(-1)
        
        # PPO-style clipping
        log_ratio = token_logprobs.sum(dim=1)
        ratio = torch.exp(log_ratio - log_ratio.mean().detach())
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.grpo_clip_epsilon, 1 + self.config.grpo_clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = -(token_logprobs * token_logprobs.exp()).sum(dim=1).mean()
        
        loss = policy_loss - self.config.grpo_entropy_coeff * entropy
        
        self.grpo_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.grpo_optimizer.step()
        
        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
        }
    
    def _compute_rewards(self, responses: List[str], prompt: str, ground_truth: str) -> List[float]:
        """计算rewards"""
        from reward_functions import compute_combined_reward
        
        rewards = []
        for response in responses:
            reward_dict = compute_combined_reward(
                response=response,
                ground_truth=ground_truth,
                question=prompt,
                weights={
                    "answer": self.config.reward_answer_weight,
                    "format": self.config.reward_format_weight,
                    "semantic": self.config.reward_semantic_weight,
                }
            )
            rewards.append(reward_dict["total"])
        
        return rewards
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """自定义collate函数"""
        max_len = max(len(item["input_ids"]) for item in batch)
        
        input_ids, attention_masks, labels = [], [], []
        
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            
            input_ids.append(torch.cat([
                item["input_ids"],
                torch.tensor([self.tokenizer.pad_token_id] * pad_len)
            ]))
            attention_masks.append(torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]))
            labels.append(torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ]))
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "prompt": [item["prompt"] for item in batch],
            "answer": [item["answer"] for item in batch],
            "difficulty": [item["difficulty"] for item in batch],
        }
