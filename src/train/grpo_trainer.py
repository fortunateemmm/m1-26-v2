"""
GRPO Trainer - 用于run_hybrid_training函数的兼容模块
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO配置"""
    num_generations: int = 8
    temperature: float = 0.7
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.01
    entropy_coeff: float = 0.01
    loss_threshold: float = 2.0
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "answer": 1.0,
        "format": 0.2,
        "semantic": 0.3,
    })


class HybridSFTGRPOTrainer:
    """
    混合SFT-GRPO训练器
    
    根据per-sample loss动态选择SFT或GRPO
    """
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        grpo_config: GRPOConfig,
        sft_optimizer,
        grpo_optimizer,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = grpo_config
        self.sft_optimizer = sft_optimizer
        self.grpo_optimizer = grpo_optimizer
        
        # 冻结参考模型
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        self.sft_steps = 0
        self.grpo_steps = 0
    
    def hybrid_step(
        self,
        input_ids,
        attention_mask,
        labels,
        prompts,
        ground_truths,
        questions,
    ) -> Dict[str, float]:
        """
        混合训练步骤
        
        根据per-sample loss选择SFT或GRPO
        """
        # 计算per-sample loss
        per_sample_loss = self._compute_per_sample_loss(input_ids, attention_mask, labels)
        
        stats = {}
        use_grpo = per_sample_loss.mean().item() >= self.config.loss_threshold
        
        if use_grpo:
            stats = self._grpo_step(input_ids, attention_mask, prompts, ground_truths)
            self.grpo_steps += 1
        else:
            stats = self._sft_step(input_ids, attention_mask, labels)
            self.sft_steps += 1
        
        stats["sft_steps"] = self.sft_steps
        stats["grpo_steps"] = self.grpo_steps
        stats["mean_loss"] = per_sample_loss.mean().item()
        stats["use_grpo"] = float(use_grpo)
        
        return stats
    
    def _compute_per_sample_loss(self, input_ids, attention_mask, labels):
        """计算per-sample loss"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # 简化：返回batch平均loss
            loss = outputs.loss.unsqueeze(0).expand(input_ids.shape[0])
        self.model.train()
        return loss
    
    def _sft_step(self, input_ids, attention_mask, labels):
        """SFT训练步骤"""
        self.model.train()
        
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
        
        return {"loss": loss.item(), "type": "sft"}
    
    def _grpo_step(self, input_ids, attention_mask, prompts, ground_truths):
        """GRPO训练步骤"""
        from reward_functions import compute_combined_reward
        
        self.model.train()
        
        device = self.model.device
        G = self.config.num_generations
        batch_size = input_ids.shape[0]
        
        # 生成G个responses
        with torch.no_grad():
            expanded_input_ids = input_ids.repeat(G, 1).to(device)
            expanded_attention_mask = attention_mask.repeat(G, 1).to(device)
            
            outputs = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=1024,
                temperature=self.config.temperature,
                do_sample=True,
                return_dict_in_generate=True,
            )
        
        sequences = outputs.sequences
        gen_tokens = sequences[:, input_ids.shape[1]:]
        
        # 解析responses
        responses = [self.tokenizer.decode(gen_tokens[i], skip_special_tokens=True) for i in range(G)]
        
        # 计算rewards
        rewards = []
        for resp in responses:
            gt = ground_truths[0] if isinstance(ground_truths, list) else ground_truths
            q = prompts[0] if isinstance(prompts, list) else prompts
            reward_dict = compute_combined_reward(resp, gt, q, self.config.reward_weights)
            rewards.append(reward_dict["total"])
        
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
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        entropy = -(token_logprobs * token_logprobs.exp()).sum(dim=1).mean()
        loss = policy_loss - self.config.entropy_coeff * entropy
        
        self.grpo_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.grpo_optimizer.step()
        
        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "type": "grpo",
        }
