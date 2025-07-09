import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from wordcloud import WordCloud
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(log_file, save_dir=None):
    """绘制训练曲线"""
    # 读取训练日志
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析日志数据
    steps = []
    losses = []
    
    for line in lines:
        if 'Step' in line and 'Loss' in line:
            parts = line.strip().split()
            step = int(parts[4].rstrip(','))
            loss = float(parts[6])
            steps.append(step)
            losses.append(loss)
    
    # 绘制曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, 'b-', alpha=0.7)
    plt.title('训练损失曲线')
    plt.xlabel('步数')
    plt.ylabel('损失')
    plt.grid(True, alpha=0.3)
    
    # 移动平均
    window_size = max(1, len(losses) // 20)
    if len(losses) > window_size:
        smoothed_losses = pd.Series(losses).rolling(window=window_size).mean()
        plt.subplot(1, 2, 2)
        plt.plot(steps, smoothed_losses, 'r-', alpha=0.8)
        plt.title(f'训练损失曲线 (移动平均, 窗口={window_size})')
        plt.xlabel('步数')
        plt.ylabel('损失')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_evaluation_results(eval_report, save_dir=None):
    """绘制评估结果"""
    if isinstance(eval_report, str):
        with open(eval_report, 'r', encoding='utf-8') as f:
            report = json.load(f)
    else:
        report = eval_report
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 基本指标柱状图
    ax1 = axes[0, 0]
    metrics = report['basic_metrics']
    names = list(metrics.keys())
    values = list(metrics.values())
    
    bars = ax1.bar(names, values, color=['blue', 'green', 'orange', 'red'])
    ax1.set_title('基本性能指标')
    ax1.set_ylabel('分数')
    ax1.set_ylim(0, 1)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 混淆矩阵
    ax2 = axes[0, 1]
    cm = np.array(report['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('混淆矩阵')
    ax2.set_xlabel('预测类别')
    ax2.set_ylabel('真实类别')
    
    # 模块统计
    if 'module_statistics' in report and report['module_statistics']:
        ax3 = axes[1, 0]
        module_names = []
        module_scores = []
        
        for name, stats in report['module_statistics'].items():
            module_names.append(name)
            if 'mean_cert_score' in stats:
                module_scores.append(stats['mean_cert_score'])
            elif 'mean_safety_score' in stats:
                module_scores.append(stats['mean_safety_score'])
            else:
                module_scores.append(1 - stats.get('mean_attack_prob', 0))
        
        ax3.bar(module_names, module_scores, color=['purple', 'cyan', 'yellow'])
        ax3.set_title('模块评估结果')
        ax3.set_ylabel('平均分数')
        ax3.set_ylim(0, 1)
    
    # 综合评估雷达图
    ax4 = axes[1, 1]
    categories = ['准确率', '精确率', '召回率', 'F1分数']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    ax4.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax4.fill(angles, values, alpha=0.25, color='blue')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('综合性能雷达图')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_distribution(features, labels, save_dir=None):
    """绘制特征分布图"""
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 5))
    
    # t-SNE可视化
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE特征分布')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # PCA可视化
    plt.subplot(1, 2, 2)
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('PCA特征分布')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()


def create_wordcloud(texts, save_dir=None):
    """创建词云图"""
    # 合并所有文本
    all_text = ' '.join(texts)
    
    # 创建词云
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path='simhei.ttf',  # 中文字体
        max_words=100,
        colormap='viridis'
    ).generate(all_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('文本词云图')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'wordcloud.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_attention_weights(attention_weights, tokens, save_dir=None):
    """绘制注意力权重热图"""
    plt.figure(figsize=(12, 8))
    
    # 创建热图
    sns.heatmap(attention_weights, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                cbar=True)
    
    plt.title('注意力权重热图')
    plt.xlabel('目标token')
    plt.ylabel('源token')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'attention_weights.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='语衡项目可视化工具')
    parser.add_argument('--type', choices=['training', 'evaluation', 'features', 'wordcloud'], 
                       required=True, help='可视化类型')
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # 根据类型进行可视化
    if args.type == 'training':
        plot_training_curves(args.input, args.output)
    elif args.type == 'evaluation':
        plot_evaluation_results(args.input, args.output)
    elif args.type == 'features':
        # 加载特征数据
        data = torch.load(args.input)
        plot_feature_distribution(data['features'], data['labels'], args.output)
    elif args.type == 'wordcloud':
        # 加载文本数据
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        create_wordcloud(texts, args.output)


if __name__ == '__main__':
    main() 