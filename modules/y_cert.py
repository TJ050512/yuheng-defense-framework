import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class YCertModule(nn.Module):
    """Y-Cert认证模块：用于检测和认证多模态内容的真实性"""
    
    def __init__(self, hidden_size=768, threshold=0.8):
        """
        初始化Y-Cert模块
        
        Args:
            hidden_size: 隐藏层大小
            threshold: 认证阈值
        """
        super(YCertModule, self).__init__()
        
        self.threshold = threshold
        self.hidden_size = hidden_size
        
        # 真实性检测网络
        self.authenticity_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 一致性检测网络
        self.consistency_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 质量评估网络
        self.quality_assessor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, image_features):
        """
        前向传播
        
        Args:
            text_features: 文本特征
            image_features: 图像特征
            
        Returns:
            Dict: 包含各种认证指标的字典
        """
        # 特征融合
        combined_features = torch.cat([text_features, image_features], dim=1)
        
        # 计算各种认证指标
        authenticity_score = self.authenticity_detector(combined_features)
        consistency_score = self.consistency_detector(combined_features)
        quality_score = self.quality_assessor(combined_features)
        
        # 计算综合认证分数
        cert_score = (authenticity_score + consistency_score + quality_score) / 3
        
        # 认证决策
        is_certified = (cert_score >= self.threshold).float()
        
        return {
            'authenticity_score': authenticity_score,
            'consistency_score': consistency_score,
            'quality_score': quality_score,
            'cert_score': cert_score,
            'is_certified': is_certified
        }
    
    def get_cert_report(self, text_features, image_features):
        """获取详细的认证报告"""
        with torch.no_grad():
            results = self.forward(text_features, image_features)
            
            # 转换为numpy数组
            report = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    report[key] = value.cpu().numpy()
                else:
                    report[key] = value
            
            return report
    
    def update_threshold(self, new_threshold: float):
        """更新认证阈值"""
        if 0 <= new_threshold <= 1:
            self.threshold = new_threshold
        else:
            raise ValueError("阈值必须在0到1之间") 