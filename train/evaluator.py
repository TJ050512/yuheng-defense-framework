import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_model import MultimodalModel
from modules.y_cert import YCertModule
from modules.attack_track import AttackTrackModule
from modules.msa import MSAModule
from data.dataset import create_dataloader


class YuhengEvaluator:
    """语衡项目评估器"""
    
    def __init__(self, model_path, config):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = None
        self.modules = {}
        self.tokenizer = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 重建模型
        model_config = self.config['model']
        self.model = MultimodalModel(
            text_encoder=model_config['text_encoder'],
            num_classes=model_config['num_classes'],
            hidden_size=model_config['hidden_size'],
            dropout=model_config['dropout'],
            fusion_method=model_config['fusion_method']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 重建模块
        modules_config = self.config['modules']
        hidden_size = model_config['hidden_size']
        
        if modules_config['y_cert']['enabled']:
            self.modules['y_cert'] = YCertModule(
                hidden_size=hidden_size,
                threshold=modules_config['y_cert']['threshold']
            ).to(self.device)
            
        if modules_config['attack_track']['enabled']:
            self.modules['attack_track'] = AttackTrackModule(
                hidden_size=hidden_size,
                detection_methods=modules_config['attack_track']['detection_methods']
            ).to(self.device)
            
        if modules_config['msa']['enabled']:
            self.modules['msa'] = MSAModule(
                hidden_size=hidden_size,
                attack_types=modules_config['msa']['attack_types']
            ).to(self.device)
        
        # 加载模块状态
        for name, module in self.modules.items():
            if name in checkpoint['modules_state_dict']:
                module.load_state_dict(checkpoint['modules_state_dict'][name])
        
        print(f"模型已从 {model_path} 加载")
    
    def evaluate_basic_metrics(self, test_loader):
        """评估基本指标"""
        self.model.eval()
        for module in self.modules.values():
            module.eval()
        
        predictions = []
        true_labels = []
        logits_list = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估基本指标"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                logits = self.model(input_ids, attention_mask, image)
                
                # 收集结果
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                logits_list.extend(logits.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels,
            'logits': logits_list
        }
    
    def evaluate_modules(self, test_loader):
        """评估扩展模块"""
        self.model.eval()
        for module in self.modules.values():
            module.eval()
        
        module_results = {name: [] for name in self.modules.keys()}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估扩展模块"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                
                # 获取特征
                features = self.model.get_features(input_ids, attention_mask, image)
                text_features = features['text_features']
                image_features = features['image_features']
                
                # 评估各模块
                for name, module in self.modules.items():
                    if name == 'y_cert':
                        results = module.get_cert_report(text_features, image_features)
                    elif name == 'attack_track':
                        results = module.get_attack_report(text_features, image_features)
                    elif name == 'msa':
                        results = module.get_safety_report(text_features, image_features)
                    
                    module_results[name].append(results)
        
        return module_results
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """绘制混淆矩阵"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_module_statistics(self, module_results, save_path=None):
        """绘制模块统计信息"""
        num_modules = len(module_results)
        if num_modules == 0:
            return
        
        fig, axes = plt.subplots(1, num_modules, figsize=(6 * num_modules, 5))
        if num_modules == 1:
            axes = [axes]
        
        for idx, (name, results) in enumerate(module_results.items()):
            ax = axes[idx]
            
            if name == 'y_cert':
                # Y-Cert模块统计
                cert_scores = np.concatenate([r['cert_score'] for r in results])
                ax.hist(cert_scores, bins=20, alpha=0.7, color='blue')
                ax.set_title(f'{name} - 认证分数分布')
                ax.set_xlabel('认证分数')
                ax.set_ylabel('频次')
                
            elif name == 'attack_track':
                # AttackTrack模块统计
                attack_probs = np.concatenate([r['max_attack_prob'] for r in results])
                ax.hist(attack_probs, bins=20, alpha=0.7, color='red')
                ax.set_title(f'{name} - 攻击概率分布')
                ax.set_xlabel('攻击概率')
                ax.set_ylabel('频次')
                
            elif name == 'msa':
                # MSA模块统计
                safety_scores = np.concatenate([r['safety_score'] for r in results])
                ax.hist(safety_scores, bins=20, alpha=0.7, color='green')
                ax.set_title(f'{name} - 安全得分分布')
                ax.set_xlabel('安全得分')
                ax.set_ylabel('频次')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, test_loader, save_path=None):
        """生成完整评估报告"""
        print("正在生成评估报告...")
        
        # 基本指标评估
        basic_metrics = self.evaluate_basic_metrics(test_loader)
        
        # 模块评估
        module_results = self.evaluate_modules(test_loader)
        
        # 生成报告
        report = {
            'basic_metrics': {
                'accuracy': float(basic_metrics['accuracy']),
                'precision': float(basic_metrics['precision']),
                'recall': float(basic_metrics['recall']),
                'f1': float(basic_metrics['f1'])
            },
            'confusion_matrix': basic_metrics['confusion_matrix'].tolist(),
            'module_statistics': {}
        }
        
        # 模块统计
        for name, results in module_results.items():
            if name == 'y_cert':
                cert_scores = np.concatenate([r['cert_score'] for r in results])
                report['module_statistics'][name] = {
                    'mean_cert_score': float(np.mean(cert_scores)),
                    'std_cert_score': float(np.std(cert_scores)),
                    'cert_rate': float(np.mean(cert_scores > 0.8))
                }
            elif name == 'attack_track':
                attack_probs = np.concatenate([r['max_attack_prob'] for r in results])
                report['module_statistics'][name] = {
                    'mean_attack_prob': float(np.mean(attack_probs)),
                    'std_attack_prob': float(np.std(attack_probs)),
                    'attack_detection_rate': float(np.mean(attack_probs > 0.5))
                }
            elif name == 'msa':
                safety_scores = np.concatenate([r['safety_score'] for r in results])
                report['module_statistics'][name] = {
                    'mean_safety_score': float(np.mean(safety_scores)),
                    'std_safety_score': float(np.std(safety_scores)),
                    'safety_rate': float(np.mean(safety_scores > 0.5))
                }
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"评估报告已保存到: {save_path}")
        
        return report, basic_metrics, module_results
    
    def test_single_sample(self, text, image_path):
        """测试单个样本"""
        # 这里需要实现单个样本的测试逻辑
        # 包括文本和图像的预处理、模型推理等
        pass 