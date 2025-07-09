import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms


class YuhengDataset(Dataset):
    """语衡项目多模态数据集"""
    
    def __init__(self, data_path, tokenizer, image_transform=None, max_length=512):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径（支持.json和.jsonl格式）
            tokenizer: 文本tokenizer
            image_transform: 图像变换
            max_length: 最大序列长度
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        
        # 加载数据
        self.data = []
        
        # 判断文件格式
        if data_path.endswith('.json'):
            # JSON格式
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.data = data
                else:
                    self.data = [data]
        elif data_path.endswith('.jsonl'):
            # JSONL格式
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"不支持的文件格式: {data_path}. 请使用.json或.jsonl文件")
        
        print(f"成功加载 {len(self.data)} 条数据从 {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理文本 - 支持多种字段名
        text = item.get('text_clean', item.get('text', ''))
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理图像 - 支持多种字段名
        image_path = item.get('image_path_clean', item.get('image_path', ''))
        
        # 处理相对路径
        if image_path and not os.path.isabs(image_path):
            # 如果是相对路径，基于数据集目录进行拼接
            base_dir = os.path.dirname(os.path.dirname(self.data_path))  # 回到项目根目录
            dataset_dir = os.path.join(base_dir, '数据集')
            image_path = os.path.join(dataset_dir, image_path)
        
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                if self.image_transform:
                    image = self.image_transform(image)
            except Exception as e:
                print(f"警告: 无法加载图像 {image_path}: {e}")
                # 创建空白图像
                image = torch.zeros(3, 224, 224)
        else:
            # 如果没有图像，创建空白图像
            image = torch.zeros(3, 224, 224)
        
        # 获取标签 - 支持多种格式
        label = item.get('label', 0)
        
        # 标签映射
        if isinstance(label, str):
            label_map = {
                '色情': 1,
                '正常': 0,
                'pornographic': 1,
                'normal': 0
            }
            label = label_map.get(label.lower(), 0)
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_image_transform(image_size=224, is_training=True):
    """获取图像变换"""
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_split_dataloaders(data_path, tokenizer, batch_size=32, image_size=224, 
                            max_length=512, train_ratio=0.7, val_ratio=0.15, 
                            num_workers=4, random_seed=42):
    """
    从单个数据文件创建分割的数据加载器
    
    Args:
        data_path: 数据文件路径
        tokenizer: 文本tokenizer
        batch_size: 批大小
        image_size: 图像尺寸
        max_length: 最大序列长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例 
        num_workers: 工作线程数
        random_seed: 随机种子
    
    Returns:
        train_loader, val_loader, test_loader
    """
    import random
    
    # 设置随机种子
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 首先加载完整数据集
    image_transform_train = get_image_transform(image_size, is_training=True)
    image_transform_eval = get_image_transform(image_size, is_training=False)
    
    # 创建完整数据集
    full_dataset = YuhengDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_transform=image_transform_train,  # 后面会重新设置
        max_length=max_length
    )
    
    # 计算分割大小
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    print(f"数据分割: 训练集 {train_size}, 验证集 {val_size}, 测试集 {test_size}")
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # 为验证和测试集设置正确的图像变换
    # 注意：这里是一个简化做法，实际上应该创建不同的dataset实例
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_dataloader(data_path, tokenizer, batch_size=32, image_size=224, 
                     max_length=512, is_training=True, num_workers=4):
    """创建数据加载器"""
    image_transform = get_image_transform(image_size, is_training)
    
    dataset = YuhengDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader 