"""
改进前后对比可视化
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def create_comparison_diagram():
    """创建改进前后的对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Original vs Improved Pipeline Comparison', fontsize=16, fontweight='bold')
    
    # 1. 数据质量分布对比
    ax1 = axes[0, 0]
    categories = ['High\nQuality', 'Medium\nQuality', 'Low\nQuality', 'Dropped']
    
    original = [70, 20, 10, 0]  # 原始方法：只保留正确答案
    improved = [60, 25, 10, 5]  # 改进方法：有质量验证
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, original, width, label='Original (SFT only)', color='#3498db')
    ax1.bar(x + width/2, improved, width, label='Improved (with verification)', color='#2ecc71')
    
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Data Quality Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 训练策略对比
    ax2 = axes[0, 1]
    
    strategies = ['Easy\nSamples', 'Medium\nSamples', 'Hard\nSamples']
    sft_pct = [100, 100, 100]  # 原始：全部SFT
    hybrid_pct = [100, 60, 30]  # 改进：部分GRPO
    
    x = np.arange(len(strategies))
    
    ax2.bar(x, sft_pct, width=0.5, label='Original (100% SFT)', color='#3498db')
    ax2.bar(x, hybrid_pct, width=0.5, label='Improved (Hybrid SFT+GRPO)', color='#e74c3c', alpha=0.8)
    ax2.bar(x, [0, 40, 70], width=0.5, bottom=hybrid_pct, label='GRPO Portion', color='#f39c12', alpha=0.8)
    
    ax2.set_ylabel('Training Strategy (%)')
    ax2.set_title('Training Strategy by Sample Difficulty')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 120)
    
    # 3. Reward维度对比
    ax3 = axes[1, 0]
    
    reward_dims = ['Answer\nCorrectness', 'Format\nCompliance', 'Semantic\nQuality', 'Self-\nVerification']
    
    original_rewards = [1.0, 0, 0, 0]  # 原始：只有答案正确性
    improved_rewards = [1.0, 0.2, 0.3, 0.2]  # 改进：多维reward
    
    x = np.arange(len(reward_dims))
    
    ax3.bar(x - width/2, original_rewards, width, label='Original', color='#3498db')
    ax3.bar(x + width/2, improved_rewards, width, label='Improved', color='#2ecc71')
    
    ax3.set_ylabel('Reward Weight')
    ax3.set_title('Reward Function Dimensions')
    ax3.set_xticks(x)
    ax3.set_xticklabels(reward_dims)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 预期效果对比
    ax4 = axes[1, 1]
    
    metrics = ['Reasoning\nAccuracy', 'Format\nCompliance', 'Semantic\nCoherence', 'Hard Sample\nImprovement']
    original_scores = [70, 80, 65, 15]
    improved_scores = [85, 95, 82, 40]
    
    x = np.arange(len(metrics))
    
    bars1 = ax4.bar(x - width/2, original_scores, width, label='Original', color='#3498db')
    bars2 = ax4.bar(x + width/2, improved_scores, width, label='Improved', color='#2ecc71')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax4.set_ylabel('Score (%)')
    ax4.set_title('Expected Performance Improvement')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('F:/vscode/m1/docs/improvement_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_pipeline_flow_diagram():
    """创建流程对比图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 原始流程
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Original Pipeline', fontsize=14, fontweight='bold', pad=20)
    
    # 原始流程框
    boxes_orig = [
        (5, 9, '196K QA Pairs'),
        (5, 7.5, 'Difficulty\nFiltering'),
        (5, 6, '37K Questions'),
        (5, 4.5, 'Thinking\nGeneration (R1)'),
        (5, 3, '23K with\nReasoning'),
        (5, 1.5, 'Diversity\nSampling'),
        (5, 0.3, '1K/23K\nDataset'),
    ]
    
    for x, y, text in boxes_orig:
        rect = mpatches.FancyBboxPatch((x-1.5, y-0.4), 3, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#e8f4f8', edgecolor='#3498db', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接箭头
    for y in [8.6, 7.1, 5.6, 4.1, 2.6, 1.1]:
        ax1.annotate('', xy=(5, y-0.5), xytext=(5, y+0.4),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    
    # SFT标注
    rect_sft = mpatches.FancyBboxPatch((7, 0.3), 2.5, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2)
    ax1.add_patch(rect_sft)
    ax1.text(8.25, 0.7, 'SFT\nOnly', ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax1.annotate('', xy=(7, 0.7), xytext=(6.5, 0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))
    
    # 改进流程
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Improved Pipeline', fontsize=14, fontweight='bold', pad=20)
    
    # 改进流程框
    boxes_improved = [
        (5, 9, '196K QA Pairs'),
        (5, 7.8, 'Difficulty\nFiltering'),
        (5, 6.6, 'Thinking\nGeneration (Multi-Teacher)'),
        (3, 5.2, 'Quality\nVerification', '#e8f5e9'),
        (7, 5.2, 'Low Quality\n→ Rejection', '#ffebee'),
        (3, 3.8, 'Difficulty-Aware\nSampling', '#e3f2fd'),
        (1.5, 2.4, 'Hard + High\nQuality', '#ffcdd2'),
        (4.5, 2.4, 'Easy + High\nQuality', '#c8e6c9'),
        (1.5, 0.8, 'GRPO\nTraining', '#ff9800'),
        (4.5, 0.8, 'SFT\nTraining', '#4caf50'),
    ]
    
    for item in boxes_improved:
        if len(item) == 4:
            x, y, text, color = item
        else:
            x, y, text = item
            color = '#e8f4f8'
        
        rect = mpatches.FancyBboxPatch((x-1.3, y-0.4), 2.6, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='#3498db', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 连接箭头
    ax2.annotate('', xy=(5, 8.3), xytext=(5, 8.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax2.annotate('', xy=(5, 7.1), xytext=(5, 7.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax2.annotate('', xy=(5, 5.7), xytext=(5, 6.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax2.annotate('', xy=(3, 5.2), xytext=(4, 5.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax2.annotate('', xy=(7, 5.2), xytext=(6, 5.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
    ax2.annotate('', xy=(3, 4.3), xytext=(3, 4.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax2.annotate('', xy=(1.5, 2.9), xytext=(2.5, 3.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax2.annotate('', xy=(4.5, 2.9), xytext=(3.5, 3.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
    ax2.annotate('', xy=(1.5, 1.3), xytext=(1.5, 1.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
    ax2.annotate('', xy=(4.5, 1.3), xytext=(4.5, 1.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2ecc71'))
    
    # 改进点标注
    improvements = [
        (8, 6.6, '① Multi-Teacher\nDistillation'),
        (8, 5.2, '② Quality\nVerification'),
        (8, 3.8, '③ Difficulty-Aware\nSampling'),
        (8, 1.6, '④ Hybrid\nSFT+GRPO'),
    ]
    
    for x, y, text in improvements:
        rect = mpatches.FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#fff9c4', edgecolor='#f39c12', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color='#e65100')
    
    plt.tight_layout()
    plt.savefig('F:/vscode/m1/docs/pipeline_flow.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Creating comparison diagrams...")
    create_comparison_diagram()
    create_pipeline_flow_diagram()
    print("Done! Check docs/ folder for images.")
