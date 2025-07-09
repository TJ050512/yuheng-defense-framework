"""
语衡项目 (Yuheng Project)

基于多模态深度学习的内容安全检测与认证系统
"""

__version__ = "1.0.0"
__author__ = "Yuheng Project Team"
__email__ = "yuheng@project.com"

from .models import MultimodalModel
from .modules import YCertModule, AttackTrackModule, MSAModule
from .train import YuhengTrainer, YuhengEvaluator
from .data import YuhengDataset, create_dataloader

__all__ = [
    'MultimodalModel',
    'YCertModule', 
    'AttackTrackModule',
    'MSAModule',
    'YuhengTrainer',
    'YuhengEvaluator',
    'YuhengDataset',
    'create_dataloader'
] 