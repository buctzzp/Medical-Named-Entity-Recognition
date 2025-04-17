标签名	含义
O	Outside，非实体（其他普通词）
B-Anatomical	实体"解剖结构"的开头（如"肺"、"胃"、"大脑"）
I-Anatomical	解剖结构实体的中间部分
B-Diseases	实体"疾病"的开头（如"糖尿病"、"肺炎"）
I-Diseases	疾病实体的中间部分
B-Drug	实体"药品"的开头（如"阿莫西林"、"布洛芬"）
I-Drug	药品实体的中间部分
B-Image	实体"影像检查术语"的开头（如"CT"、"MRI"、"超声"）
I-Image	影像实体的中间部分
B-Laboratory	实体"实验室指标/检查"的开头（如"血糖"、"血压"、"肌酐"）
I-Laboratory	实验室实体的中间部分
B-Operation	实体"操作/手术名词"的开头（如"剖宫产"、"关节置换"）
I-Operation	操作实体的中间部分

本模型使用了 BIO 标注体系，共识别 6 类医学实体，包括 Anatomical（解剖结构）、Diseases（疾病）、Drug（药品）、Image（影像检查）、Laboratory（实验室指标）和 Operation（手术操作）。每类标签分为 B（开始）和 I（内部），共计 13 个标签。

# 中文医疗命名实体识别 (NER) 系统

基于BERT系列预训练模型的中文医疗命名实体识别系统，支持多种模型架构和优化方法。

## 目录

- [项目概述](#项目概述)
- [主要特性](#主要特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据格式](#数据格式)
- [配置文件](#配置文件)
- [使用说明](#使用说明)
  - [训练模型](#训练模型)
  - [预测实体](#预测实体)
  - [评估模型](#评估模型)
- [模型架构](#模型架构)
- [数据增强](#数据增强)
- [模型优化](#模型优化)
- [结果可视化](#结果可视化)
- [常见问题](#常见问题)
- [示例](#示例)

## 项目概述

本项目实现了一个基于深度学习的中文医疗命名实体识别系统，能够从医疗文本中提取关键实体（如疾病、症状、药物、治疗方法等）。项目支持多种预训练模型和优化技术，适用于医疗领域的信息抽取任务。

## 主要特性

- **多种预训练模型**：支持多种中文预训练模型，包括通用和医疗领域专用模型
- **灵活的模型架构**：支持BERT-CRF、BERT-Attention-CRF以及BiLSTM等多种架构组合
- **增强版训练**：支持数据增强、早停、梯度裁剪等训练优化技术
- **增强版预测**：支持批量处理、交互模式、多种输出格式
- **模型优化**：支持模型剪枝、量化等优化技术，提高推理效率
- **结果可视化**：提供实体分布可视化和注意力可视化
- **模型可解释性**：支持注意力分析和特征归因

## 项目结构

```
.
├── config.py                  # 配置文件
├── train_enhanced.py          # 增强版训练脚本
├── predict_enhanced.py        # 增强版预测脚本
├── evaluate.py                # 模型评估脚本
├── data_augmentation.py       # 数据增强工具
├── model_explainability.py    # 模型可解释性工具
├── model_optimization.py      # 模型优化工具
├── utils.py                   # 工具函数
├── model/                     # 模型定义
│   ├── model_factory.py       # 模型工厂
│   ├── bert_crf_model.py      # BERT-CRF模型定义
│   └── bert_attention_crf.py  # BERT-Attention-CRF模型定义
├── data/                      # 数据目录
│   ├── train.txt              # 训练数据
│   ├── dev.txt                # 开发集数据
│   └── test.txt               # 测试数据
├── dicts/                     # 词典目录
│   ├── synonyms.json          # 同义词词典
│   └── entity_dict.json       # 实体词典
├── logs/                      # 日志目录
├── models/                    # 模型保存目录
├── results/                   # 结果输出目录
└── README.md                  # 本文档
```

## 环境配置

### 依赖安装

```bash
# 创建并激活虚拟环境（可选）
python -m venv venv
source venv/bin/activate   # Linux/MacOS
# 或者在Windows上：
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 依赖列表 (requirements.txt)

```
torch>=1.7.0
transformers>=4.5.0
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.50.0
seqeval>=1.2.2
```

## 数据格式

系统使用BIO标注格式（Beginning, Inside, Outside）的文本文件：

```
患 O
者 O
出 O
现 O
高 B-Disease
血 I-Disease
压 I-Disease
和 O
2 B-Disease
型 I-Disease
糖 I-Disease
尿 I-Disease
病 I-Disease
```

每行包含一个字符和一个标签，标签用空格分隔，不同样本之间用空行分隔。

支持的实体类型（可在配置中自定义）：
- Disease（疾病）
- Symptom（症状）
- Drug（药物）
- Treatment（治疗）
- Test（检查）
- Anatomy（解剖部位）
- ...（可扩展）

## 配置文件

`config.py`文件包含系统的默认配置参数，可通过命令行参数覆盖。主要配置项如下：

```python
# 模型配置
model_config = {
    'pretrained_model_name': 'bert-base-chinese',  # 预训练模型名称
    'max_len': 128,                                # 最大序列长度
    'batch_size': 16,                              # 训练批次大小
    'eval_batch_size': 32,                         # 评估批次大小
    'num_epochs': 10,                              # 训练轮数
    'learning_rate': 2e-5,                         # 学习率
    'weight_decay': 0.01,                          # 权重衰减
    'warmup_ratio': 0.1,                           # 预热比例
    'max_grad_norm': 1.0,                          # 梯度裁剪
    'early_stopping_patience': 3,                  # 早停耐心值
    'logging_steps': 10,                           # 日志记录间隔

    # 数据路径
    'train_path': 'data/train.txt',                # 训练数据路径
    'dev_path': 'data/dev.txt',                    # 开发集路径
    'test_path': 'data/test.txt',                  # 测试集路径

    # 模型保存路径
    'model_dir': 'models',                         # 模型保存目录
    'log_dir': 'logs',                             # 日志保存目录
    'best_model_path': 'models/best_model.pth',    # 最佳模型路径
    'final_model_path': 'models/final_model.pth',  # 最终模型路径

    # LSTM配置
    'use_bilstm': True,                            # 是否使用BiLSTM
    'lstm_hidden_size': 128,                       # LSTM隐藏层大小
    'lstm_layers': 1,                              # LSTM层数
    'lstm_dropout': 0.1,                           # LSTM丢弃率

    # 注意力配置
    'use_attention': False,                        # 是否使用注意力
    'attention_size': 128,                         # 注意力大小
    'num_attention_heads': 8,                      # 注意力头数
    'attention_dropout': 0.1,                      # 注意力丢弃率
    'hidden_dropout': 0.1,                         # 隐藏层丢弃率
}

# 数据增强配置
augmentation_config = {
    'use_data_augmentation': False,                # 是否使用数据增强
    'synonym_dict_path': 'dicts/synonyms.json',    # 同义词词典路径
    'entity_dict_path': 'dicts/entity_dict.json',  # 实体词典路径
    'use_synonym_replace': True,                   # 使用同义词替换
    'use_entity_replace': True,                    # 使用实体替换
    'synonym_replace_ratio': 0.2,                  # 同义词替换比例
    'entity_replace_ratio': 0.15,                  # 实体替换比例
}

# 模型优化配置
optimization_config = {
    'use_model_pruning': False,                    # 是否使用模型剪枝
    'use_model_quantization': False,               # 是否使用模型量化
    'pruning_method': 'magnitude',                 # 剪枝方法
    'pruning_ratio': 0.3,                          # 剪枝比例
    'quantization_method': 'dynamic',              # 量化方法
    'quantization_dtype': 'qint8',                 # 量化数据类型
}

# 模型可解释性配置
explainability_config = {
    'visualization_output_dir': 'results/attention_viz',  # 可视化输出目录
    'attention_layer_indices': [-1, -2, -3, -4],          # 注意力层索引
    'attention_head_indices': [0, 1, 2, 3],               # 注意力头索引
}

# 实验追踪配置
experiment_config = {
    'use_wandb': False,                            # 是否使用Weights & Biases
    'project_name': 'chinese-medical-ner',         # 项目名称
}
```

## 使用说明

### 训练模型

使用`train_enhanced.py`脚本训练模型：

```bash
python train_enhanced.py --pretrained_model bert-base-chinese --use_bilstm --batch_size 16 --epochs 10
```

主要参数说明：

- `--pretrained_model`: 预训练模型名称，支持 bert-base-chinese、chinese-medical-bert、pcl-medbert、cmeee-bert、mc-bert、chinese-roberta-med 等
- `--use_attention`: 是否使用注意力机制
- `--use_bilstm`: 是否使用BiLSTM层
- `--no_bilstm`: 禁用BiLSTM层（覆盖默认配置）
- `--batch_size`: 训练批次大小
- `--epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--use_augmentation`: 启用数据增强
- `--no_augmentation`: 禁用数据增强（覆盖默认配置）
- `--early_stopping`: 早停耐心值
- `--save_every_epoch`: 是否每个epoch保存模型
- `--lstm_hidden_size`: LSTM隐藏层大小
- `--lstm_layers`: LSTM层数
- `--use_model_pruning`: 是否使用模型剪枝
- `--use_model_quantization`: 是否使用模型量化
- `--seed`: 随机种子

### 预测实体

使用`predict_enhanced.py`脚本进行预测：

```bash
# 从文件批量预测
python predict_enhanced.py --input data/samples.txt --output results/predictions.json --pretrained_model bert-base-chinese --use_bilstm --model best

# 单个文本预测
python predict_enhanced.py --input "患者出现高血压和2型糖尿病，建议服用降压药。" --pretrained_model bert-base-chinese --use_bilstm --model best

# 交互式预测（不提供input参数）
python predict_enhanced.py --pretrained_model bert-base-chinese --use_bilstm --model best
```

主要参数说明：

- `--input`: 输入文件路径或文本，每行一个句子，不提供则进入交互模式
- `--output`: 输出文件路径，不提供则自动生成
- `--model`: 模型权重路径，可选 'best'/'final'/'quantized'/'pruned'或具体路径
- `--format`: 输出格式，可选 'json'/'text'/'bio'
- `--batch_size`: 批处理大小
- `--pretrained_model`: 预训练模型名称
- `--use_attention`: 是否使用注意力模型
- `--use_bilstm`: 是否使用BiLSTM层
- `--no_bilstm`: 禁用BiLSTM层（覆盖默认配置）
- `--label_path`: 标签映射文件路径
- `--max_length`: 最大序列长度
- `--pretty`: 美化JSON输出
- `--detailed`: 输出详细的实体识别信息，包括位置和类型概率

### 评估模型

使用`evaluate.py`脚本评估模型性能（假设该脚本存在）：

```bash
python evaluate.py --model models/best_model_bert_base_chinese_base_with_bilstm.pth --pretrained_model bert-base-chinese --use_bilstm
```

## 模型架构

项目支持多种模型架构组合：

### BERT-CRF

BERT + 条件随机场(CRF)模型，适合序列标注任务，能捕获标签之间的依赖关系。

### BERT-BiLSTM-CRF

BERT + 双向LSTM + CRF模型，BiLSTM可以进一步捕获上下文信息，对长距离依赖有更好的建模能力。

### BERT-Attention-CRF

BERT + 自注意力机制 + CRF模型，增加自注意力层可以更好地捕获不同位置的token之间的关系。

### BERT-Attention-BiLSTM-CRF

完整版模型，结合了所有组件的优势，通常能获得更好的性能，但计算开销也最大。

## 数据增强

通过`data_augmentation.py`模块，项目支持多种数据增强方法：

### 同义词替换

根据同义词词典，随机替换文本中的某些词语为其同义词，增加语义多样性。

```python
# 数据增强配置示例
augmentation_config = {
    'use_synonym_replace': True,       # 启用同义词替换
    'synonym_replace_ratio': 0.2,      # 替换比例
    'synonym_dict_path': 'dicts/synonyms.json'  # 同义词词典
}
```

### 实体替换

根据实体词典，随机替换文本中的实体为同类型的其他实体，增加实体多样性。

```python
augmentation_config = {
    'use_entity_replace': True,        # 启用实体替换
    'entity_replace_ratio': 0.15,      # 替换比例
    'entity_dict_path': 'dicts/entity_dict.json'  # 实体词典
}
```

## 模型优化

项目支持通过`model_optimization.py`模块进行模型优化：

### 模型剪枝

通过剪枝减少模型参数，降低模型大小，加快推理速度。

```bash
python train_enhanced.py --use_model_pruning --pretrained_model bert-base-chinese
```

### 模型量化

通过量化减少参数精度，降低内存占用和计算开销。

```bash
python train_enhanced.py --use_model_quantization --pretrained_model bert-base-chinese
```

## 结果可视化

### 实体分布可视化

预测时如果处理多个样本，会自动生成实体类型分布图表：

```bash
python predict_enhanced.py --input data/samples.txt --pretrained_model bert-base-chinese
```

图表将保存在输出文件同目录下，命名为 `[output_filename]_entity_distribution.png`。

### 注意力可视化

分析模型注意力分布，了解模型关注的文本区域（通过`model_explainability.py`）。

## 常见问题

### 1. 模型加载失败

**问题**: 运行预测时提示"模型加载失败"。

**解决方案**:
- 确保使用与训练时相同的预训练模型和架构配置
- 检查模型路径是否正确
- 如果使用自定义模型路径，确保该路径下存在模型文件

### 2. CUDA内存不足

**问题**: 训练时提示CUDA内存不足。

**解决方案**:
- 减小批次大小 (--batch_size)
- 减小序列最大长度 (config.py中的max_len)
- 如果使用BiLSTM，考虑减小隐藏层大小或禁用BiLSTM

### 3. 预测结果不理想

**问题**: 模型预测的实体不准确。

**解决方案**:
- 检查训练数据质量和标注一致性
- 增加训练数据量或使用数据增强
- 尝试不同的预训练模型（如医疗领域专用模型）
- 调整模型架构（添加BiLSTM或注意力机制）

## 示例

### 训练示例

```bash
# 使用bert-base-chinese，带BiLSTM，启用数据增强
python train_enhanced.py --pretrained_model bert-base-chinese --use_bilstm --use_augmentation --batch_size 16 --epochs 15

# 使用医疗领域预训练模型，带注意力机制和BiLSTM
python train_enhanced.py --pretrained_model chinese-medical-bert --use_attention --use_bilstm --batch_size 8 --epochs 10
```

### 预测示例

```bash
# 批量预测，输出JSON格式
python predict_enhanced.py --input data/test_samples.txt --format json --pretrained_model bert-base-chinese --use_bilstm --model best

# 单条文本预测，输出文本格式
python predict_enhanced.py --input "患者男，45岁，因高血压3年，近期头痛加剧就诊。" --format text --pretrained_model bert-base-chinese --use_bilstm
```

预测输出示例 (JSON格式)：
```json
[
  {
    "text": "患者男，45岁，因高血压3年，近期头痛加剧就诊。",
    "entities": [
      {
        "type": "Disease",
        "text": "高血压",
        "start": 8,
        "end": 11
      },
      {
        "type": "Symptom",
        "text": "头痛",
        "start": 15,
        "end": 17
      }
    ]
  }
]
```

预测输出示例 (文本格式)：
```
患者男，45岁，因[高血压:Disease]3年，近期[头痛:Symptom]加剧就诊。
```

### 评估示例

```bash
# 评估模型性能
python evaluate.py --model models/best_model_chinese_medical_bert_attention_with_bilstm.pth --pretrained_model chinese-medical-bert --use_attention --use_bilstm
```
