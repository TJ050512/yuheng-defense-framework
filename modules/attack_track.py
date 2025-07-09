import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class AttackTrackModule(nn.Module):
    """AttackTrack攻击检测模块：用于检测和追踪各种类型的攻击"""
    
    def __init__(self, hidden_size=768, detection_methods=None):
        """
        初始化AttackTrack模块
        
        Args:
            hidden_size: 隐藏层大小
            detection_methods: 检测方法列表
        """
        super(AttackTrackModule, self).__init__()
        
        self.hidden_size = hidden_size
        self.detection_methods = detection_methods or ["adversarial", "backdoor", "poisoning"]
        
        # 对抗样本检测器
        self.adversarial_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 后门攻击检测器
        self.backdoor_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 数据投毒检测器
        self.poisoning_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 攻击类型分类器
        self.attack_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, len(self.detection_methods)),
            nn.Softmax(dim=1)
        )
        
        # 攻击强度评估器
        self.intensity_assessor = nn.Sequential(
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
            Dict: 包含各种攻击检测结果的字典
        """
        # 特征融合
        combined_features = torch.cat([text_features, image_features], dim=1)
        
        # 攻击检测
        adversarial_prob = self.adversarial_detector(combined_features)
        backdoor_prob = self.backdoor_detector(combined_features)
        poisoning_prob = self.poisoning_detector(combined_features)
        
        # 攻击类型分类
        attack_type_probs = self.attack_classifier(combined_features)
        
        # 攻击强度评估
        attack_intensity = self.intensity_assessor(combined_features)
        
        # 综合攻击概率
        max_attack_prob = torch.max(torch.stack([
            adversarial_prob, backdoor_prob, poisoning_prob
        ], dim=1), dim=1)[0]
        
        return {
            'adversarial_prob': adversarial_prob,
            'backdoor_prob': backdoor_prob,
            'poisoning_prob': poisoning_prob,
            'attack_type_probs': attack_type_probs,
            'attack_intensity': attack_intensity,
            'max_attack_prob': max_attack_prob,
            'is_attack': (max_attack_prob > 0.5).float()
        }
    
    def get_attack_report(self, text_features, image_features):
        """获取详细的攻击检测报告"""
        with torch.no_grad():
            results = self.forward(text_features, image_features)
            
            # 转换为numpy数组
            report = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    report[key] = value.cpu().numpy()
                else:
                    report[key] = value
            
            # 添加攻击类型预测
            attack_type_idx = np.argmax(report['attack_type_probs'], axis=1)
            report['predicted_attack_type'] = [self.detection_methods[idx] for idx in attack_type_idx]
            
            return report
    
    def track_attack_sequence(self, feature_sequence: List[Tuple[torch.Tensor, torch.Tensor]]):
        """追踪攻击序列"""
        attack_history = []
        
        for text_features, image_features in feature_sequence:
            results = self.forward(text_features, image_features)
            attack_history.append(results)
        
        return attack_history
    
    def analyze_attack_pattern(self, attack_history: List[Dict]):
        """分析攻击模式"""
        if not attack_history:
            return {}
        
        # 统计攻击频率
        attack_counts = sum(result['is_attack'].sum().item() for result in attack_history)
        total_samples = len(attack_history) * attack_history[0]['is_attack'].size(0)
        attack_frequency = attack_counts / total_samples
        
        # 分析攻击强度变化
        intensities = [result['attack_intensity'].mean().item() for result in attack_history]
        intensity_trend = np.polyfit(range(len(intensities)), intensities, 1)[0]
        
        return {
            'attack_frequency': attack_frequency,
            'average_intensity': np.mean(intensities),
            'intensity_trend': intensity_trend,
            'total_attacks': attack_counts,
            'total_samples': total_samples
        } 