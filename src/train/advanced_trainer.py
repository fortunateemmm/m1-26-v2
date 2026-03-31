"""
高级训练策略

包含：
1. 课程学习（Curriculum Learning）
2. 自适应Loss阈值
3. 困难样本回放
4. GRPO温度退火
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import json
import os
import math

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """课程学习阶段配置"""
    name: str
    difficulty_range: Tuple[float, float]
    epochs: int
    grpo_ratio: float = 0.3  # GRPO样本比例


@dataclass
class AdvancedTrainingConfig:
    """高级训练配置"""
    
    # 基础参数
    learning_rate: float = 1e-5
    grpo_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    total_epochs: int = 5
    
    # 动态Loss阈值
    initial_loss_threshold: float = 3.0  # 初始阈值（较高，少用GRPO）
    final_loss_threshold: float = 1.5    # 最终阈值（较低，多用GRPO）
    threshold_annealing: bool = True     # 是否启用阈值退火
    
    # GRPO参数
    initial_grpo_temperature: float = 0.9  # 初始温度（高探索）
    final_grpo_temperature: float = 0.5    # 最终温度（低探索）
    grpo_num_generations: int = 8
    grpo_clip_epsilon: float = 0.2
    grpo_entropy_coeff: float = 0.01
    
    # 奖励权重
    reward_answer_weight: float = 1.0
    reward_format_weight: float = 0.2
    reward_semantic_weight: float = 0.3
    
    # 课程学习
    use_curriculum: bool = True
    curriculum_stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage("warmup", (0.3, 0.5), epochs=1, grpo_ratio=0.1),
        CurriculumStage("main_easy", (0.4, 0.6), epochs=1, grpo_ratio=0.2),
        CurriculumStage("main_medium", (0.5, 0.7), epochs=2, grpo_ratio=0.3),
        CurriculumStage("main_hard", (0.6, 0.8), epochs=2, grpo_ratio=0.4),
        CurriculumStage("final_hard", (0.7, 1.0), epochs=2, grpo_ratio=0.5),
    ])
    
    # 困难样本回放
    use_hard_sample_replay: bool = True
    replay_buffer_size: int = 500
    replay_ratio: float = 0.2  # 每个batch中回放样本的比例
    
    # 数据路径
    data_path: Optional[str] = None


class ReplayBuffer:
    """
    困难样本回放缓冲区
    
    保存loss较高的样本，在后续epoch中反复训练
    """
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.loss_stats = {}  # sample_id -> loss history
    
    def add(self, sample: Dict, loss: float):
        """添加样本到缓冲区"""
        sample_id = id(sample)  # 或使用其他唯一标识
        
        if sample_id not in self.loss_stats:
            self.loss_stats[sample_id] = []
        self.loss_stats[sample_id].append(loss)
        
        # 计算平均loss
        avg_loss = sum(self.loss_stats[sample_id]) / len(self.loss_stats[sample_id])
        
        # 保存样本
        self.buffer.append({
            "sample": sample,
            "avg_loss": avg_loss,
            "visit_count": len(self.loss_stats[sample_id]),
        })
    
    def sample(self, n: int) -> List[Dict]:
        """从缓冲区采样"""
        if len(self.buffer) == 0:
            return []
        
        # 按avg_loss加权采样（loss越高越可能被采样）
        losses = [item["avg_loss"] for item in self.buffer]
        total = sum(losses)
        
        if total == 0:
            indices = torch.randint(0, len(self.buffer), (n,)).tolist()
        else:
            probs = [l / total for l in losses]
            indices = torch.multinomial(torch.tensor(probs), min(n, len(self.buffer))).tolist()
        
        return [self.buffer[i]["sample"] for i in indices]
    
    def get_stats(self) -> Dict:
        """获取缓冲区统计"""
        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "avg_loss": sum(item["avg_loss"] for item in self.buffer) / len(self.buffer) if self.buffer else 0,
        }


class CurriculumScheduler:
    """
    课程学习调度器
    
    按照难度从易到难训练
    """
    
    def __init__(self, stages: List[CurriculumStage]):
        self.stages = stages
        self.current_stage = 0
        self.current_epoch = 0
    
    def get_stage(self) -> CurriculumStage:
        """获取当前阶段配置"""
        if self.current_stage >= len(self.stages):
            return self.stages[-1]
        return self.stages[self.current_stage]
    
    def should_advance(self, metrics: Dict) -> bool:
        """判断是否进入下一阶段"""
        current_stage = self.get_stage()
        
        # 检查epoch数
        epochs_in_stage = metrics.get("epochs_in_stage", 0)
        if epochs_in_stage < current_stage.epochs:
            return False
        
        # 检查loss是否收敛
        avg_loss = metrics.get("avg_loss", float("inf"))
        if avg_loss < 0.5:
            return True
        
        # 默认：完成epoch数后进入下一阶段
        return True
    
    def advance(self):
        """进入下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.current_epoch = 0
            stage = self.get_stage()
            logger.info(f"\n{'='*50}")
            logger.info(f"Curriculum: Entering stage '{stage.name}'")
            logger.info(f"  Difficulty range: {stage.difficulty_range}")
            logger.info(f"  Epochs: {stage.epochs}")
            logger.info(f"  GRPO ratio: {stage.grpo_ratio}")
            logger.info(f"{'='*50}\n")
    
    def filter_by_difficulty(self, samples: List[Dict]) -> List[Dict]:
        """根据当前阶段筛选样本"""
        stage = self.get_stage()
        low, high = stage.difficulty_range
        
        filtered = [
            s for s in samples
            if low <= s.get("difficulty", 0.5) < high
        ]
        
        logger.info(f"Filtered {len(filtered)} samples for stage '{stage.name}' (difficulty {low}-{high})")
        return filtered
    
    def get_grpo_ratio(self) -> float:
        """获取当前阶段的GRPO比例"""
        return self.get_stage().grpo_ratio


class AdaptiveThresholdScheduler:
    """
    自适应Loss阈值调度器
    
    根据训练进度动态调整loss阈值：
    - 初期：阈值较高，只对真正困难的样本用GRPO
    - 后期：阈值较低，更多样本用GRPO精调
    """
    
    def __init__(
        self,
        initial_threshold: float = 3.0,
        final_threshold: float = 1.5,
        annealing_type: str = "cosine",  # "linear", "cosine", "exponential"
    ):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.annealing_type = annealing_type
    
    def get_threshold(self, progress: float) -> float:
        """
        获取当前阈值
        
        Args:
            progress: 训练进度 (0-1)
        """
        if self.annealing_type == "linear":
            # 线性退火
            return self.initial_threshold + (self.final_threshold - self.initial_threshold) * progress
        elif self.annealing_type == "cosine":
            # 余弦退火
            return self.final_threshold + 0.5 * (self.initial_threshold - self.final_threshold) * (1 + math.cos(math.pi * progress))
        elif self.annealing_type == "exponential":
            # 指数退火
            return self.initial_threshold * math.exp(-3 * progress)
        else:
            return self.initial_threshold


class TemperatureScheduler:
    """
    GRPO温度退火调度器
    
    训练后期降低采样温度，让模型收敛
    """
    
    def __init__(
        self,
        initial_temp: float = 0.9,
        final_temp: float = 0.5,
        annealing_type: str = "cosine",
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.annealing_type = annealing_type
    
    def get_temperature(self, progress: float) -> float:
        """获取当前温度"""
        if self.annealing_type == "cosine":
            return self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (1 + math.cos(math.pi * progress))
        else:
            return self.initial_temp + (self.final_temp - self.initial_temp) * progress


class AdvancedHybridTrainer:
    """
    高级混合训练器
    
    集成所有高级训练策略：
    1. 课程学习
    2. 自适应Loss阈值
    3. 困难样本回放
    4. GRPO温度退火
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: AdvancedTrainingConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 优化器
        self.sft_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.grpo_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.grpo_learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 调度器
        self.curriculum = CurriculumScheduler(config.curriculum_stages) if config.use_curriculum else None
        self.threshold_scheduler = AdaptiveThresholdScheduler(
            initial_threshold=config.initial_loss_threshold,
            final_threshold=config.final_loss_threshold,
        )
        self.temperature_scheduler = TemperatureScheduler(
            initial_temp=config.initial_grpo_temperature,
            final_temp=config.final_grpo_temperature,
        )
        
        # 困难样本回放
        self.replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size) if config.use_hard_sample_replay else None
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.total_steps = 0
        
        # 统计
        self.stats = {
            "sft_steps": 0,
            "grpo_steps": 0,
            "replay_samples": 0,
        }
    
    def train(
        self,
        train_dataset: Dataset,
        batch_size: int = 1,
    ):
        """执行训练"""
        all_samples = list(train_dataset)
        
        logger.info(f"Starting advanced hybrid training with {len(all_samples)} samples")
        logger.info(f"  Curriculum learning: {self.config.use_curriculum}")
        logger.info(f"  Hard sample replay: {self.config.use_hard_sample_replay}")
        logger.info(f"  Adaptive threshold: {self.config.threshold_annealing}")
        
        # 课程学习阶段
        if self.curriculum:
            for stage_idx, stage in enumerate(self.config.curriculum_stages):
                logger.info(f"\n{'='*50}")
                logger.info(f"Curriculum Stage {stage_idx + 1}/{len(self.config.curriculum_stages)}: {stage.name}")
                logger.info(f"{'='*50}")
                
                # 筛选当前阶段的样本
                stage_samples = self.curriculum.filter_by_difficulty(all_samples)
                
                if not stage_samples:
                    logger.warning(f"No samples for stage {stage.name}, skipping...")
                    self.curriculum.advance()
                    continue
                
                # 训练当前阶段
                for epoch in range(stage.epochs):
                    self._train_epoch(
                        samples=stage_samples,
                        batch_size=batch_size,
                        grpo_ratio=stage.grpo_ratio,
                        stage_progress=epoch / stage.epochs,
                    )
                
                self.curriculum.advance()
        else:
            # 不使用课程学习，直接训练所有样本
            for epoch in range(self.config.total_epochs):
                progress = epoch / self.config.total_epochs
                self._train_epoch(
                    samples=all_samples,
                    batch_size=batch_size,
                    grpo_ratio=0.3,
                    stage_progress=progress,
                )
        
        # 打印最终统计
        self._print_final_stats()
    
    def _train_epoch(
        self,
        samples: List[Dict],
        batch_size: int,
        grpo_ratio: float,
        stage_progress: float = 0.0,
    ):
        """训练一个epoch"""
        self.current_epoch += 1
        
        # 计算当前参数
        overall_progress = (self.current_epoch - 1) / max(self.config.total_epochs, 1)
        current_threshold = self.threshold_scheduler.get_threshold(overall_progress)
        current_temp = self.temperature_scheduler.get_temperature(overall_progress)
        
        logger.info(f"\nEpoch {self.current_epoch}:")
        logger.info(f"  Loss threshold: {current_threshold:.3f}")
        logger.info(f"  GRPO temperature: {current_temp:.3f}")
        logger.info(f"  GRPO ratio: {grpo_ratio:.2f}")
        
        # 创建dataloader
        dataset = UnifiedDataset(samples, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        
        epoch_stats = {"sft_loss": [], "grpo_loss": [], "hard_count": 0, "easy_count": 0}
        
        for batch_idx, batch in enumerate(loader):
            # 判断使用SFT还是GRPO
            use_grpo = self._should_use_grpo(batch, current_threshold)
            
            if use_grpo:
                stats = self._grpo_step(batch, current_temp)
                epoch_stats["grpo_loss"].append(stats.get("loss", 0))
                epoch_stats["hard_count"] += 1
                self.stats["grpo_steps"] += 1
                
                # 添加到回放缓冲区
                if self.replay_buffer:
                    self.replay_buffer.add(batch, stats.get("loss", 0))
            else:
                stats = self._sft_step(batch)
                epoch_stats["sft_loss"].append(stats.get("loss", 0))
                epoch_stats["easy_count"] += 1
                self.stats["sft_steps"] += 1
            
            self.global_step += 1
            
            # 插入回放样本
            if self.replay_buffer and self.global_step % 10 == 0:
                replay_samples = self.replay_buffer.sample(int(batch_size * self.config.replay_ratio))
                if replay_samples:
                    self._train_replay(replay_samples, current_temp)
                    self.stats["replay_samples"] += len(replay_samples)
            
            # 日志
            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"  Step {self.global_step} | "
                    f"{'GRPO' if use_grpo else 'SFT'} | "
                    f"Loss: {stats.get('loss', 0):.4f} | "
                    f"Temp: {current_temp:.2f} | "
                    f"Threshold: {current_threshold:.2f}"
                )
        
        # Epoch统计
        logger.info(f"\nEpoch {self.current_epoch} Summary:")
        logger.info(f"  Easy samples (SFT): {epoch_stats['easy_count']}")
        logger.info(f"  Hard samples (GRPO): {epoch_stats['hard_count']}")
        if epoch_stats['sft_loss']:
            logger.info(f"  Avg SFT loss: {sum(epoch_stats['sft_loss'])/len(epoch_stats['sft_loss']):.4f}")
        if epoch_stats['grpo_loss']:
            logger.info(f"  Avg GRPO loss: {sum(epoch_stats['grpo_loss'])/len(epoch_stats['grpo_loss']):.4f}")
        if self.replay_buffer:
            logger.info(f"  Replay buffer: {self.replay_buffer.get_stats()}")
    
    def _should_use_grpo(self, batch: Dict, threshold: float) -> bool:
        """判断是否使用GRPO"""
        # 基于difficulty预判
        difficulty = batch["difficulty"][0] if isinstance(batch["difficulty"], list) else batch["difficulty"]
        
        # 高难度直接用GRPO
        if difficulty > 0.8:
            return True
        
        # 低难度直接用SFT
        if difficulty < 0.4:
            return False
        
        # 中等难度：基于实时loss判断
        return self._compute_loss_and_check(batch, threshold)
    
    def _compute_loss_and_check(self, batch: Dict, threshold: float) -> bool:
        """计算loss并判断"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.item()
        
        self.model.train()
        return loss >= threshold
    
    def _sft_step(self, batch: Dict) -> Dict[str, float]:
        """SFT训练步骤"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        labels = batch["labels"].to(self.model.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        self.sft_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.sft_optimizer.step()
        
        return {"loss": loss.item()}
    
    def _grpo_step(self, batch: Dict, temperature: float) -> Dict[str, float]:
        """GRPO训练步骤"""
        self.model.train()
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
        answer = batch["answer"][0] if isinstance(batch["answer"], list) else batch["answer"]
        
        device = self.model.device
        G = self.config.grpo_num_generations
        
        # 生成G个responses
        with torch.no_grad():
            expanded_input_ids = input_ids.repeat(G, 1).to(device)
            expanded_attention_mask = attention_mask.repeat(G, 1).to(device)
            
            outputs = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
            )
        
        sequences = outputs.sequences
        gen_tokens = sequences[:, input_ids.shape[1]:]
        responses = [self.tokenizer.decode(gen_tokens[i], skip_special_tokens=True) for i in range(G)]
        
        # 计算rewards
        rewards = self._compute_rewards(responses, prompt, answer)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # GRPO loss
        logits = self.model(sequences).logits[:, -gen_tokens.shape[1]-1:-1, :]
        token_logprobs = F.log_softmax(logits, dim=-1)
        token_logprobs = token_logprobs.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Group-relative advantages
        rewards_grouped = rewards.view(1, G)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_grouped - group_mean) / group_std).view(-1)
        
        log_ratio = token_logprobs.sum(dim=1)
        ratio = torch.exp(log_ratio - log_ratio.mean().detach())
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.grpo_clip_epsilon, 1 + self.config.grpo_clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        entropy = -(token_logprobs * token_logprobs.exp()).sum(dim=1).mean()
        loss = policy_loss - self.config.grpo_entropy_coeff * entropy
        
        self.grpo_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.grpo_optimizer.step()
        
        return {"loss": loss.item(), "mean_reward": rewards.mean().item()}
    
    def _train_replay(self, samples: List[Dict], temperature: float):
        """训练回放样本"""
        for sample in samples:
            # 转换为batch格式
            batch = {
                "input_ids": sample["input_ids"].unsqueeze(0),
                "attention_mask": sample["attention_mask"].unsqueeze(0),
                "labels": sample["labels"].unsqueeze(0),
                "prompt": [sample.get("prompt", "")],
                "answer": [sample.get("answer", "")],
            }
            
            # 对回放样本使用GRPO（因为它们是困难样本）
            self._grpo_step(batch, temperature)
    
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
        """Collate function"""
        max_len = max(len(item["input_ids"]) for item in batch)
        
        input_ids, attention_masks, labels = [], [], []
        
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(torch.cat([item["input_ids"], torch.tensor([self.tokenizer.pad_token_id] * pad_len)]))
            attention_masks.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "prompt": [item.get("prompt", "") for item in batch],
            "answer": [item.get("answer", "") for item in batch],
            "difficulty": [item.get("difficulty", 0.5) for item in batch],
        }
    
    def _print_final_stats(self):
        """打印最终统计"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Complete - Final Statistics")
        logger.info(f"{'='*50}")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"SFT steps: {self.stats['sft_steps']}")
        logger.info(f"GRPO steps: {self.stats['grpo_steps']}")
        logger.info(f"Replay samples: {self.stats['replay_samples']}")
        logger.info(f"{'='*50}\n")


class UnifiedDataset(Dataset):
    """统一数据集"""
    
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 32768):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        prompt = sample.get("prompt", sample.get("question", ""))
        reasoning = sample.get("reasoning", "")
        answer = sample.get("answer", "")
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n{answer}"},
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
        
        labels = encoding["input_ids"].copy()
        assistant_token = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        
        for i in range(len(labels) - len(assistant_token)):
            if encoding["input_ids"][i:i+len(assistant_token)] == assistant_token:
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
