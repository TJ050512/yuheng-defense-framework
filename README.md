# 语衡项目 (Yuheng Project)

语衡项目是一个基于多模态深度学习的内容安全检测与认证系统，集成了BERT+ResNet+MLP架构以及多个专业安全模块。

## 项目结构

```
yuheng_project/
├── config/             # 配置文件
│   └── default.yaml    # 默认配置
├── data/               # 数据处理模块
│   ├── __init__.py
│   └── dataset.py      # 数据集定义
├── models/             # 模型定义
│   ├── __init__.py
│   └── multimodal_model.py  # 多模态模型
├── modules/            # 扩展模块
│   ├── __init__.py
│   ├── y_cert.py       # Y-Cert认证模块
│   ├── attack_track.py # AttackTrack攻击检测模块
│   └── msa.py          # MSA多模态安全分析模块
├── train/              # 训练与评估
│   ├── __init__.py
│   ├── trainer.py      # 训练器
│   └── evaluator.py    # 评估器
├── scripts/            # 工具脚本
│   ├── visualize.py    # 可视化工具
│   └── data_conversion.py  # 数据转换工具
├── logs/               # 训练日志
├── checkpoints/        # 模型检查点
├── run.py              # 主运行入口
├── requirements.txt    # 依赖列表
└── README.md          # 项目说明
```

## 主要功能

### 1. 多模态模型
- **BERT文本编码器**: 处理中文文本内容
- **ResNet图像编码器**: 提取图像特征
- **MLP分类器**: 融合多模态特征进行分类
- **多种融合方式**: 支持concat、attention、cross_attention

### 2. 安全模块
- **Y-Cert认证模块**: 内容真实性检测与认证
- **AttackTrack模块**: 攻击检测与追踪
- **MSA安全分析模块**: 综合安全风险评估

### 3. 工具与脚本
- **可视化工具**: 训练曲线、评估结果、特征分布等
- **数据转换**: 支持CSV、JSON到JSONL格式转换
- **模型评估**: 全面的性能评估与报告生成

## 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 数据准备
数据格式要求为JSONL，每行包含：
```json
{
    "text": "文本内容",
    "image_path": "图像路径",
    "label": 0
}
```

使用数据转换工具：
```bash
python scripts/data_conversion.py --task csv2jsonl --input data.csv --output data.jsonl
```

### 3. 训练模型
```bash
python run.py --mode train --config config/default.yaml
```

### 4. 评估模型
```bash
python run.py --mode eval --config config/default.yaml --model_path checkpoints/best_model.pth --test_data data/test.jsonl
```

### 5. 单样本预测
```bash
python run.py --mode predict --config config/default.yaml --model_path checkpoints/best_model.pth --text "测试文本" --image "test.jpg"
```

## 配置说明

主要配置参数位于`config/default.yaml`：

```yaml
# 模型配置
model:
  name: "yuheng_multimodal"
  text_encoder: "bert-base-chinese"
  image_encoder: "resnet50"
  hidden_size: 768
  num_classes: 2
  dropout: 0.1
  fusion_method: "concat"

# 训练配置
training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 500

# 模块配置
modules:
  y_cert:
    enabled: true
    threshold: 0.8
  attack_track:
    enabled: true
    detection_methods: ["adversarial", "backdoor"]
  msa:
    enabled: true
    attack_types: ["text", "image", "multimodal"]
```

## 可视化工具

### 训练曲线
```bash
python scripts/visualize.py --type training --input logs/train.log --output visualizations/
```

### 评估结果
```bash
python scripts/visualize.py --type evaluation --input evaluation_report.json --output visualizations/
```

### 特征分布
```bash
python scripts/visualize.py --type features --input features.pt --output visualizations/
```

## 模块详解

### Y-Cert认证模块
提供内容真实性检测功能：
- 真实性评分
- 一致性检测
- 质量评估
- 综合认证决策

### AttackTrack攻击检测模块
检测和追踪各种攻击：
- 对抗样本检测
- 后门攻击识别
- 数据投毒检测
- 攻击模式分析

### MSA多模态安全分析模块
综合安全风险评估：
- 安全风险分级
- 有害内容检测
- 误导性内容识别
- 隐私泄露检测

## 性能指标

系统提供多种评估指标：
- 基础指标：准确率、精确率、召回率、F1分数
- 模块指标：认证率、攻击检测率、安全评分
- 可视化：混淆矩阵、ROC曲线、特征分布

## 开发指南

### 添加新模块
1. 在`modules/`目录创建新模块文件
2. 继承`nn.Module`基类
3. 实现前向传播和评估方法
4. 在`modules/__init__.py`中注册

### 自定义数据集
1. 继承`YuhengDataset`类
2. 实现`__getitem__`方法
3. 处理特定的数据格式

### 模型改进
1. 修改`models/multimodal_model.py`
2. 调整网络结构或融合方式
3. 更新配置文件

## 注意事项

1. 确保CUDA环境正确配置
2. 图像路径需要使用绝对路径或相对于项目根目录的路径
3. 中文文本处理需要合适的tokenizer
4. 训练过程中建议定期保存检查点

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。 