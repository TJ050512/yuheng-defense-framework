#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语衡项目主运行入口
"""

import argparse
import os
import sys
import yaml
import torch
from transformers import AutoTokenizer

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from train.trainer import YuhengTrainer
from train.evaluator import YuhengEvaluator
from models.multimodal_model import MultimodalModel
from data.dataset import create_dataloader


def train_model(config_path):
    """训练模型"""
    print("开始训练模型...")
    trainer = YuhengTrainer(config_path)
    trainer.train()
    print("训练完成!")


def evaluate_model(config_path, model_path, test_data_path):
    """评估模型"""
    print("开始评估模型...")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建评估器
    evaluator = YuhengEvaluator(model_path, config)
    
    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    
    # 创建测试数据加载器
    test_loader = create_dataloader(
        data_path=test_data_path,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        max_length=config['data']['max_seq_length'],
        is_training=False,
        num_workers=config['data']['num_workers']
    )
    
    # 生成评估报告
    report_path = os.path.join(config['output_dir'], 'evaluation_report.json')
    report, basic_metrics, module_results = evaluator.generate_report(test_loader, report_path)
    
    print("评估完成!")
    print(f"准确率: {basic_metrics['accuracy']:.4f}")
    print(f"F1分数: {basic_metrics['f1']:.4f}")


def predict_single(config_path, model_path, text, image_path):
    """单样本预测"""
    print("开始单样本预测...")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = MultimodalModel(
        text_encoder=config['model']['text_encoder'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout'],
        fusion_method=config['model']['fusion_method']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder'])
    
    # 数据预处理
    # 这里需要实现单样本的预处理逻辑
    # ...
    
    print("预测完成!")


def main():
    parser = argparse.ArgumentParser(description='语衡项目主程序')
    parser.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True,
                       help='运行模式')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--model_path', help='模型路径（用于评估和预测）')
    parser.add_argument('--test_data', help='测试数据路径（用于评估）')
    parser.add_argument('--text', help='输入文本（用于预测）')
    parser.add_argument('--image', help='输入图像路径（用于预测）')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        return
    
    # 根据模式执行相应功能
    if args.mode == 'train':
        train_model(args.config)
    elif args.mode == 'eval':
        if not args.model_path or not args.test_data:
            print("错误: 评估模式需要指定 --model_path 和 --test_data")
            return
        evaluate_model(args.config, args.model_path, args.test_data)
    elif args.mode == 'predict':
        if not args.model_path or not args.text or not args.image:
            print("错误: 预测模式需要指定 --model_path, --text 和 --image")
            return
        predict_single(args.config, args.model_path, args.text, args.image)


if __name__ == '__main__':
    main() 