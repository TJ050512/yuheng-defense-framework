#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换脚本
用于将各种格式的数据转换为项目所需的标准格式
"""

import json
import os
import argparse
from PIL import Image
import pandas as pd
from tqdm import tqdm


def convert_csv_to_jsonl(csv_path, output_path, text_column, image_column, label_column):
    """
    将CSV格式数据转换为JSONL格式
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出JSONL文件路径
        text_column: 文本列名
        image_column: 图像路径列名
        label_column: 标签列名
    """
    print(f"正在转换 {csv_path} 到 {output_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换数据
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="转换进度"):
            item = {
                'text': str(row[text_column]) if not pd.isna(row[text_column]) else '',
                'image_path': str(row[image_column]) if not pd.isna(row[image_column]) else '',
                'label': int(row[label_column]) if not pd.isna(row[label_column]) else 0
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成，共处理 {len(df)} 条数据")


def convert_json_to_jsonl(json_path, output_path, text_key='text', image_key='image', label_key='label'):
    """
    将JSON格式数据转换为JSONL格式
    
    Args:
        json_path: JSON文件路径
        output_path: 输出JSONL文件路径
        text_key: 文本字段名
        image_key: 图像路径字段名
        label_key: 标签字段名
    """
    print(f"正在转换 {json_path} 到 {output_path}")
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换数据
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="转换进度"):
            converted_item = {
                'text': item.get(text_key, ''),
                'image_path': item.get(image_key, ''),
                'label': item.get(label_key, 0)
            }
            f.write(json.dumps(converted_item, ensure_ascii=False) + '\n')
    
    print(f"转换完成，共处理 {len(data)} 条数据")


def validate_images(jsonl_path, image_base_path=None):
    """
    验证JSONL文件中的图像路径是否有效
    
    Args:
        jsonl_path: JSONL文件路径
        image_base_path: 图像基础路径（如果图像路径是相对路径）
    """
    print(f"正在验证 {jsonl_path} 中的图像...")
    
    valid_count = 0
    invalid_count = 0
    invalid_items = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="验证进度"), 1):
            item = json.loads(line.strip())
            image_path = item.get('image_path', '')
            
            if not image_path:
                continue
            
            # 处理相对路径
            if image_base_path and not os.path.isabs(image_path):
                full_path = os.path.join(image_base_path, image_path)
            else:
                full_path = image_path
            
            # 检查文件是否存在
            if os.path.exists(full_path):
                try:
                    # 尝试打开图像
                    with Image.open(full_path) as img:
                        img.verify()
                    valid_count += 1
                except Exception as e:
                    invalid_count += 1
                    invalid_items.append({
                        'line': line_num,
                        'path': full_path,
                        'error': str(e)
                    })
            else:
                invalid_count += 1
                invalid_items.append({
                    'line': line_num,
                    'path': full_path,
                    'error': '文件不存在'
                })
    
    print(f"验证完成:")
    print(f"  有效图像: {valid_count}")
    print(f"  无效图像: {invalid_count}")
    
    if invalid_items:
        print("无效图像列表:")
        for item in invalid_items[:10]:  # 只显示前10个
            print(f"  行 {item['line']}: {item['path']} - {item['error']}")
        if len(invalid_items) > 10:
            print(f"  ... 还有 {len(invalid_items) - 10} 个无效图像")


def split_dataset(jsonl_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    Args:
        jsonl_path: JSONL文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    import random
    
    print(f"正在分割数据集 {jsonl_path}")
    
    # 读取数据
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # 随机打乱
    random.seed(random_seed)
    random.shuffle(data)
    
    # 计算分割点
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # 分割数据
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # 保存分割后的数据
    base_path = os.path.splitext(jsonl_path)[0]
    
    # 训练集
    train_path = f"{base_path}_train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 验证集
    val_path = f"{base_path}_val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 测试集
    test_path = f"{base_path}_test.jsonl"
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"分割完成:")
    print(f"  训练集: {len(train_data)} 条 -> {train_path}")
    print(f"  验证集: {len(val_data)} 条 -> {val_path}")
    print(f"  测试集: {len(test_data)} 条 -> {test_path}")


def main():
    parser = argparse.ArgumentParser(description='数据格式转换工具')
    parser.add_argument('--task', choices=['csv2jsonl', 'json2jsonl', 'validate', 'split'], 
                       required=True, help='转换任务')
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--text_column', default='text', help='文本列名')
    parser.add_argument('--image_column', default='image_path', help='图像路径列名')
    parser.add_argument('--label_column', default='label', help='标签列名')
    parser.add_argument('--image_base_path', help='图像基础路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    if args.task == 'csv2jsonl':
        if not args.output:
            print("错误: csv2jsonl 任务需要指定 --output")
            return
        convert_csv_to_jsonl(args.input, args.output, args.text_column, 
                           args.image_column, args.label_column)
    
    elif args.task == 'json2jsonl':
        if not args.output:
            print("错误: json2jsonl 任务需要指定 --output")
            return
        convert_json_to_jsonl(args.input, args.output, args.text_column, 
                            args.image_column, args.label_column)
    
    elif args.task == 'validate':
        validate_images(args.input, args.image_base_path)
    
    elif args.task == 'split':
        split_dataset(args.input, args.train_ratio, args.val_ratio, 
                     args.test_ratio, args.random_seed)


if __name__ == '__main__':
    main() 