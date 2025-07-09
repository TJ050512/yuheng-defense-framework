import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml
import os
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_model import MultimodalModel
from modules.y_cert import YCertModule
from modules.attack_track import AttackTrackModule
from modules.msa import MSAModule
from data.dataset import create_dataloader


class YuhengTrainer:
    """语衡项目训练器"""
    
    def __init__(self, config_path):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 初始化模型
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # 初始化模块
        self.modules = {}
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['log_dir'], 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def build_model(self):
        """构建模型"""
        model_config = self.config['model']
        
        # 创建主模型
        self.model = MultimodalModel(
            text_encoder=model_config['text_encoder'],
            num_classes=model_config['num_classes'],
            hidden_size=model_config['hidden_size'],
            dropout=model_config['dropout'],
            fusion_method=model_config['fusion_method']
        ).to(self.device)
        
        # 创建tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['text_encoder'])
        
        # 创建扩展模块
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
        
        self.logger.info(f"模型构建完成，参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
    def build_optimizer(self):
        """构建优化器和调度器"""
        train_config = self.config['training']
        
        # 收集所有参数
        params = list(self.model.parameters())
        for module in self.modules.values():
            params.extend(module.parameters())
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            params,
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # 创建学习率调度器
        total_steps = train_config['num_epochs'] * len(self.train_loader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=train_config['warmup_steps'],
            num_training_steps=total_steps
        )
        
    def create_dataloaders(self):
        """创建数据加载器"""
        data_config = self.config['data']
        training_config = self.config['training']
        
        # 检查是否所有数据路径都相同，如果是则使用分割数据功能
        if (data_config['train_path'] == data_config['val_path'] == data_config['test_path']):
            self.logger.info("检测到使用同一数据文件，将自动分割数据集...")
            
            from data.dataset import create_split_dataloaders
            
            # 获取分割比例配置
            split_ratios = data_config.get('split_ratios', {})
            train_ratio = split_ratios.get('train', 0.7)
            val_ratio = split_ratios.get('val', 0.15)
            test_ratio = split_ratios.get('test', 0.15)
            
            # 确保比例总和为1
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                self.logger.warning(f"数据分割比例总和不为1 ({total_ratio})，将自动调整")
                train_ratio = train_ratio / total_ratio
                val_ratio = val_ratio / total_ratio
                test_ratio = test_ratio / total_ratio
            
            self.logger.info(f"数据分割比例 - 训练集: {train_ratio:.1%}, 验证集: {val_ratio:.1%}, 测试集: {test_ratio:.1%}")
            
            self.train_loader, self.val_loader, self.test_loader = create_split_dataloaders(
                data_path=data_config['train_path'],
                tokenizer=self.tokenizer,
                batch_size=training_config['batch_size'],
                image_size=data_config['image_size'],
                max_length=data_config['max_seq_length'],
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                num_workers=data_config['num_workers'],
                random_seed=self.config.get('seed', 42)
            )
        else:
            # 使用独立的数据文件
            self.train_loader = create_dataloader(
                data_path=data_config['train_path'],
                tokenizer=self.tokenizer,
                batch_size=training_config['batch_size'],
                image_size=data_config['image_size'],
                max_length=data_config['max_seq_length'],
                is_training=True,
                num_workers=data_config['num_workers']
            )
            
            self.val_loader = create_dataloader(
                data_path=data_config['val_path'],
                tokenizer=self.tokenizer,
                batch_size=training_config['batch_size'],
                image_size=data_config['image_size'],
                max_length=data_config['max_seq_length'],
                is_training=False,
                num_workers=data_config['num_workers']
            )
        
        self.logger.info(f"数据加载器创建完成，训练样本: {len(self.train_loader.dataset)}, 验证样本: {len(self.val_loader.dataset)}")
        
        # 如果有测试集，也记录其大小
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            self.logger.info(f"测试样本: {len(self.test_loader.dataset)}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        for module in self.modules.values():
            module.train()
        
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            image = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 主模型预测
            logits = self.model(input_ids, attention_mask, image)
            main_loss = nn.CrossEntropyLoss()(logits, labels)
            
            # 获取特征用于模块训练
            features = self.model.get_features(input_ids, attention_mask, image)
            text_features = features['text_features']
            image_features = features['image_features']
            
            # 模块损失
            module_losses = {}
            if 'y_cert' in self.modules:
                # Y-Cert模块损失（这里需要根据实际任务设计）
                cert_results = self.modules['y_cert'](text_features, image_features)
                module_losses['y_cert'] = torch.mean(cert_results['cert_score'])
            
            if 'attack_track' in self.modules:
                # AttackTrack模块损失
                attack_results = self.modules['attack_track'](text_features, image_features)
                module_losses['attack_track'] = torch.mean(attack_results['max_attack_prob'])
            
            if 'msa' in self.modules:
                # MSA模块损失
                msa_results = self.modules['msa'](text_features, image_features)
                module_losses['msa'] = torch.mean(msa_results['safety_score'])
            
            # 总损失
            total_loss_batch = main_loss
            for loss in module_losses.values():
                total_loss_batch += 0.1 * loss  # 模块损失权重
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + [p for module in self.modules.values() for p in module.parameters()],
                self.config['training']['max_grad_norm']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += total_loss_batch.item()
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': total_loss_batch.item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # 日志记录
            if self.global_step % self.config['training']['logging_steps'] == 0:
                self.logger.info(f"Step {self.global_step}, Loss: {total_loss_batch.item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        for module in self.modules.values():
            module.eval()
        
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                logits = self.model(input_ids, attention_mask, image)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                total_loss += loss.item()
                
                # 收集预测结果
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        avg_loss = total_loss / len(self.val_loader)
        
        metrics = {
            'eval_loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def test(self):
        """测试模型"""
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            self.logger.warning("没有测试数据集，跳过测试")
            return None
            
        self.logger.info("开始测试模型...")
        self.model.eval()
        for module in self.modules.values():
            module.eval()
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                logits = self.model(input_ids, attention_mask, image)
                
                # 收集预测结果
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        test_metrics = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }
        
        self.logger.info("测试完成！")
        self.logger.info(f"测试准确率: {accuracy:.4f}")
        self.logger.info(f"测试精确率: {precision:.4f}")
        self.logger.info(f"测试召回率: {recall:.4f}")
        self.logger.info(f"测试F1分数: {f1:.4f}")
        
        return test_metrics
    
    def save_model(self, save_path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'modules_state_dict': {name: module.state_dict() for name, module in self.modules.items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': self.epoch,
            'global_step': self.global_step
        }, save_path)
        
        self.logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        for name, module in self.modules.items():
            if name in checkpoint['modules_state_dict']:
                module.load_state_dict(checkpoint['modules_state_dict'][name])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        self.logger.info(f"模型已从 {load_path} 加载")
    
    def train(self):
        """开始训练"""
        self.logger.info("开始训练...")
        
        # 构建模型
        self.build_model()
        
        # 创建数据加载器
        self.create_dataloaders()
        
        # 构建优化器
        self.build_optimizer()
        
        # 训练循环
        best_f1 = 0
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 评估
            if (epoch + 1) % self.config['training']['eval_steps'] == 0:
                eval_metrics = self.evaluate()
                
                self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
                self.logger.info(f"Train Loss: {train_loss:.4f}")
                for key, value in eval_metrics.items():
                    self.logger.info(f"{key}: {value:.4f}")
                
                # 保存最佳模型
                if eval_metrics['f1'] > best_f1:
                    best_f1 = eval_metrics['f1']
                    best_model_path = os.path.join(self.config['output_dir'], 'best_model.pth')
                    self.save_model(best_model_path)
            
            # 定期保存检查点
            if (epoch + 1) % self.config['training']['save_steps'] == 0:
                checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_model(checkpoint_path)
        
        self.logger.info("训练完成!")
        
        # 保存最终模型
        final_model_path = os.path.join(self.config['output_dir'], 'final_model.pth')
        self.save_model(final_model_path)
        
        # 使用最佳模型进行测试
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            best_model_path = os.path.join(self.config['output_dir'], 'best_model.pth')
            if os.path.exists(best_model_path):
                self.logger.info("加载最佳模型进行测试...")
                self.load_model(best_model_path)
            
            test_metrics = self.test()
            if test_metrics:
                self.logger.info("最终测试结果:")
                for key, value in test_metrics.items():
                    self.logger.info(f"{key}: {value:.4f}") 