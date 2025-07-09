import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class MSAModule(nn.Module):
    """MSA（多模态安全分析）模块：综合分析多模态内容的安全性"""
    
    def __init__(self, hidden_size=768, attack_types=None):
        """
        初始化MSA模块
        
        Args:
            hidden_size: 隐藏层大小
            attack_types: 攻击类型列表
        """
        super(MSAModule, self).__init__()
        
        self.hidden_size = hidden_size
        self.attack_types = attack_types or ["text", "image", "multimodal"]
        
        # 安全风险评估器
        self.risk_assessor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 5),  # 5个风险级别
            nn.Softmax(dim=1)
        )
        
        # 内容有害性检测器
        self.harmful_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 误导性检测器
        self.misleading_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 隐私泄露检测器
        self.privacy_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 攻击类型分析器
        self.attack_analyzer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, len(self.attack_types)),
            nn.Softmax(dim=1)
        )
        
        # 安全置信度评估器
        self.confidence_assessor = nn.Sequential(
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
            Dict: 包含各种安全分析结果的字典
        """
        # 特征融合
        combined_features = torch.cat([text_features, image_features], dim=1)
        
        # 安全风险评估
        risk_levels = self.risk_assessor(combined_features)
        
        # 各种检测
        harmful_prob = self.harmful_detector(combined_features)
        misleading_prob = self.misleading_detector(combined_features)
        privacy_prob = self.privacy_detector(combined_features)
        
        # 攻击类型分析
        attack_type_probs = self.attack_analyzer(combined_features)
        
        # 安全置信度
        safety_confidence = self.confidence_assessor(combined_features)
        
        # 综合安全得分
        safety_score = 1 - torch.max(torch.stack([
            harmful_prob, misleading_prob, privacy_prob
        ], dim=1), dim=1)[0]
        
        # 风险级别预测
        risk_level = torch.argmax(risk_levels, dim=1)
        
        return {
            'risk_levels': risk_levels,
            'risk_level': risk_level,
            'harmful_prob': harmful_prob,
            'misleading_prob': misleading_prob,
            'privacy_prob': privacy_prob,
            'attack_type_probs': attack_type_probs,
            'safety_confidence': safety_confidence,
            'safety_score': safety_score,
            'is_safe': (safety_score > 0.5).float()
        }
    
    def get_safety_report(self, text_features, image_features):
        """获取详细的安全分析报告"""
        with torch.no_grad():
            results = self.forward(text_features, image_features)
            
            # 转换为numpy数组
            report = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    report[key] = value.cpu().numpy()
                else:
                    report[key] = value
            
            # 添加风险级别描述
            risk_descriptions = ["很低", "低", "中等", "高", "很高"]
            report['risk_description'] = [risk_descriptions[level] for level in report['risk_level']]
            
            # 添加攻击类型预测
            attack_type_idx = np.argmax(report['attack_type_probs'], axis=1)
            report['predicted_attack_type'] = [self.attack_types[idx] for idx in attack_type_idx]
            
            return report
    
    def analyze_content_safety(self, content_batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        """批量分析内容安全性"""
        safety_reports = []
        
        for text_features, image_features in content_batch:
            report = self.get_safety_report(text_features, image_features)
            safety_reports.append(report)
        
        return safety_reports
    
    def get_safety_summary(self, safety_reports: List[Dict]):
        """获取安全性摘要"""
        if not safety_reports:
            return {}
        
        # 统计安全性指标
        total_samples = len(safety_reports)
        safe_count = sum(1 for report in safety_reports if report['is_safe'].mean() > 0.5)
        
        # 计算平均风险级别
        avg_risk_level = np.mean([report['risk_level'].mean() for report in safety_reports])
        
        # 计算各类风险的平均概率
        avg_harmful_prob = np.mean([report['harmful_prob'].mean() for report in safety_reports])
        avg_misleading_prob = np.mean([report['misleading_prob'].mean() for report in safety_reports])
        avg_privacy_prob = np.mean([report['privacy_prob'].mean() for report in safety_reports])
        
        # 计算平均安全得分
        avg_safety_score = np.mean([report['safety_score'].mean() for report in safety_reports])
        
        return {
            'total_samples': total_samples,
            'safe_count': safe_count,
            'safety_rate': safe_count / total_samples,
            'avg_risk_level': avg_risk_level,
            'avg_harmful_prob': avg_harmful_prob,
            'avg_misleading_prob': avg_misleading_prob,
            'avg_privacy_prob': avg_privacy_prob,
            'avg_safety_score': avg_safety_score
        } 