import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models


class MultimodalModel(nn.Module):
    """BERT+ResNet+MLP多模态模型"""
    
    def __init__(self, text_encoder='bert-base-chinese', num_classes=2, 
                 hidden_size=768, dropout=0.1, fusion_method='concat'):
        """
        初始化多模态模型
        
        Args:
            text_encoder: 文本编码器名称
            num_classes: 分类数量
            hidden_size: 隐藏层大小
            dropout: dropout率
            fusion_method: 融合方法 ('concat', 'attention', 'cross_attention')
        """
        super(MultimodalModel, self).__init__()
        
        self.fusion_method = fusion_method
        
        # 文本编码器 (BERT)
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.text_hidden_size = self.text_encoder.config.hidden_size
        
        # 图像编码器 (ResNet)
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()  # 移除最后的分类层
        self.image_hidden_size = 2048
        
        # 特征投影层
        self.text_projection = nn.Linear(self.text_hidden_size, hidden_size)
        self.image_projection = nn.Linear(self.image_hidden_size, hidden_size)
        
        # 融合层
        if fusion_method == 'concat':
            self.fusion_input_size = hidden_size * 2
        elif fusion_method == 'attention':
            self.fusion_input_size = hidden_size
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        elif fusion_method == 'cross_attention':
            self.fusion_input_size = hidden_size
            self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
        # MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.text_projection, self.image_projection, self.classifier]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask, image):
        """
        前向传播
        
        Args:
            input_ids: 文本token ids
            attention_mask: 注意力掩码
            image: 图像张量
            
        Returns:
            logits: 分类logits
        """
        # 文本编码
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_features = self.text_projection(text_features)
        
        # 图像编码
        image_features = self.image_encoder(image)
        image_features = self.image_projection(image_features)
        
        # 特征融合
        if self.fusion_method == 'concat':
            fused_features = torch.cat([text_features, image_features], dim=1)
        elif self.fusion_method == 'attention':
            # 自注意力融合
            combined_features = torch.stack([text_features, image_features], dim=1)
            attended_features, _ = self.attention(
                combined_features, combined_features, combined_features
            )
            fused_features = attended_features.mean(dim=1)
        elif self.fusion_method == 'cross_attention':
            # 交叉注意力融合
            text_features_expanded = text_features.unsqueeze(1)
            image_features_expanded = image_features.unsqueeze(1)
            
            cross_attended, _ = self.cross_attention(
                text_features_expanded, image_features_expanded, image_features_expanded
            )
            fused_features = cross_attended.squeeze(1)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_features(self, input_ids, attention_mask, image):
        """获取特征表示（用于分析）"""
        with torch.no_grad():
            # 文本编码
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_features = text_outputs.last_hidden_state[:, 0, :]
            text_features = self.text_projection(text_features)
            
            # 图像编码
            image_features = self.image_encoder(image)
            image_features = self.image_projection(image_features)
            
            return {
                'text_features': text_features,
                'image_features': image_features
            } 