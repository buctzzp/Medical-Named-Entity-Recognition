# evaluate.py 增强版评估脚本，支持命令行参数、混淆矩阵和批量评估

import os
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from model.bert_crf_model import BertCRF
from model.bert_attention_crf_model import BertAttentionCRF
from utils import NERDataset, get_label_list, read_data_from_file, filter_special_tokens
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from config import model_config
from model.model_factory import ModelFactory
import json
from datetime import datetime

# 命令行参数解析
parser = argparse.ArgumentParser(description='中文医疗NER模型评估工具')
parser.add_argument('--data', type=str, default=model_config['test_path'], help='要评估的数据集路径')
parser.add_argument('--model', type=str, default=model_config['best_model_path'], help='模型路径')
parser.add_argument('--batch_size', type=int, default=model_config['eval_batch_size'], help='评估批次大小')
parser.add_argument('--output', type=str, default='results/evaluation', help='结果输出目录')
parser.add_argument('--confusion_matrix', action='store_true', help='是否生成混淆矩阵')
parser.add_argument('--pretrained_model', type=str, default='bert-base-chinese', 
                    help='预训练模型名称，可选：bert-base-chinese, chinese-medical-bert, pcl-medbert, cmeee-bert, mc-bert, chinese-roberta-med')
parser.add_argument('--use_attention', action='store_true', help='是否使用注意力模型')
parser.add_argument('--use_bilstm', action='store_true', help='是否使用BiLSTM层')
args = parser.parse_args()

# 创建必要的目录
os.makedirs(args.output, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.output, 'evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 记录评估配置信息
logger.info("="*50)
logger.info("医学命名实体识别模型评估开始")
logger.info("="*50)
logger.info(f"评估数据集: {args.data}")
logger.info(f"模型路径: {args.model}")
logger.info(f"预训练模型: {args.pretrained_model}")
logger.info(f"批次大小: {args.batch_size}")
logger.info(f"结果输出目录: {args.output}")
logger.info(f"是否生成混淆矩阵: {args.confusion_matrix}")
logger.info("="*50)

# 检测GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 使用ModelFactory获取与指定预训练模型匹配的tokenizer
tokenizer = ModelFactory.get_tokenizer_for_model(args.pretrained_model)

# 加载数据集
logger.info("加载评估数据集...")
test_texts, test_tags = read_data_from_file(args.data)
logger.info(f"Loaded {len(test_texts)} evaluation samples")

# 构建标签映射
label_list = get_label_list([args.data])
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label_list)

logger.info(f"标签总数: {num_labels}")
logger.info(f"标签列表: {label_list}")

# 加载模型
logger.info(f"加载模型: {args.model}")
# 根据参数选择模型类型
if args.use_attention:
    model = BertAttentionCRF.from_pretrained(
        args.pretrained_model,
        num_labels=num_labels,
        use_bilstm=args.use_bilstm,
        lstm_hidden_size=model_config['lstm_hidden_size'],
        lstm_layers=model_config['lstm_layers'],
        lstm_dropout=model_config['lstm_dropout'],
        attention_size=model_config['attention_size'],
        num_attention_heads=model_config['num_attention_heads'],
        attention_dropout=model_config['attention_dropout'],
        hidden_dropout=model_config['hidden_dropout']
    )
else:
    model = BertCRF.from_pretrained(
        args.pretrained_model,
        num_labels=num_labels,
        use_bilstm=args.use_bilstm,
        lstm_hidden_size=model_config['lstm_hidden_size'],
        lstm_layers=model_config['lstm_layers'],
        lstm_dropout=model_config['lstm_dropout']
    )

# 加载预训练权重
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)
model.eval()

logger.info("开始评估...")

# 用于收集所有的真实标签和预测标签
all_true_labels = []
all_pred_labels = []
samples_with_errors = []
all_words = []

with torch.no_grad():
    for batch in tqdm(test_texts, desc="评估进度"):
        # 从数据集中获取原始文本和标签
        words = batch
        input_ids = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=model_config['max_len']
        )["input_ids"].to(device)
        attention_mask = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=model_config['max_len']
        )["attention_mask"].to(device)
        true_label_ids = [label2id[tag] for tag in test_tags if tag in label2id]
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取预测结果
        pred_label_ids = outputs.argmax(dim=2).cpu().numpy()[0]
        
        # 处理结果 (移除padding，并转换回标签文本)
        for i in range(len(true_label_ids)):
            true_seq = [id2label[id] for id in true_label_ids[i] if id != -100]
            pred_seq = [id2label[id] for id, true_id in zip(pred_label_ids[i], true_label_ids[i]) if true_id != -100]
            
            if len(true_seq) != len(pred_seq):
                # 这种情况不应该发生，但为了安全起见
                logger.warning(f"长度不匹配: 真实标签长度 {len(true_seq)}, 预测标签长度 {len(pred_seq)}")
                continue
            
            all_true_labels.append(true_seq)
            all_pred_labels.append(pred_seq)
            
            # 收集单词
            word_seq = words[i][:len(true_seq)]
            all_words.append(word_seq)
            
            # 收集包含错误的样本
            if true_seq != pred_seq:
                errors = []
                for w, t, p in zip(word_seq, true_seq, pred_seq):
                    if t != p:
                        errors.append(f"{w} [真: {t}, 预: {p}]")
                samples_with_errors.append({
                    'words': ' '.join(word_seq),
                    'true_labels': ' '.join(true_seq),
                    'pred_labels': ' '.join(pred_seq),
                    'errors': ' | '.join(errors)
                })

# 计算评估指标
precision = precision_score(all_true_labels, all_pred_labels)
recall = recall_score(all_true_labels, all_pred_labels)
f1 = f1_score(all_true_labels, all_pred_labels)
report = classification_report(all_true_labels, all_pred_labels)

# 输出结果
result_text = f"""
评估结果:
数据集: {args.data}
模型: {args.model}
预训练模型: {args.pretrained_model}
模型类型: {'BERT-Attention-CRF' if args.use_attention else 'BERT-CRF'}
BiLSTM: {'启用' if args.use_bilstm else '禁用'}

样本数: {len(all_true_labels)}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}

分类报告:
{report}
"""

logger.info(result_text)

# 保存结果到文件
with open(os.path.join(args.output, 'classification_report.json'), 'w', encoding='utf-8') as f:
    json.dump(classification_report(all_true_labels, all_pred_labels, digits=4, output_dict=True), f, ensure_ascii=False, indent=4)
logger.info("分类报告已保存到 classification_report.json")

# 保存预测结果
predictions_path = os.path.join(args.output, 'predictions.txt')
with open(predictions_path, 'w', encoding='utf-8') as f:
    for words, true_labels, pred_labels in zip(all_words, all_true_labels, all_pred_labels):
        for w, t, p in zip(words, true_labels, pred_labels):
            f.write(f"{w} {t} {p}\n")
        f.write("\n")

logger.info(f"预测结果已保存到: {predictions_path}")

# 如果需要，创建混淆矩阵
if args.confusion_matrix:
    logger.info("生成混淆矩阵...")
    
    # 准备数据
    unique_labels = sorted(list(set([label for labels in all_true_labels for label in labels])))
    true_flattened = [label for labels in all_true_labels for label in labels]
    pred_flattened = [label for labels in all_pred_labels for label in labels]
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_flattened, pred_flattened, labels=unique_labels)
    
    # 使用Seaborn创建热图
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=unique_labels, 
        yticklabels=unique_labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # 保存混淆矩阵图
    cm_path = os.path.join(args.output, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"混淆矩阵已保存到: {cm_path}")
    
    # 为可读性，也保存为CSV格式
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm_csv_path = os.path.join(args.output, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path)
    logger.info(f"混淆矩阵CSV格式已保存到: {cm_csv_path}")

# 保存评估指标
metrics = {
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'model': args.pretrained_model,
    'model_path': args.model,
    'model_type': 'bert-attention-crf' if args.use_attention else 'bert-crf',
    'use_bilstm': args.use_bilstm,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'entity_metrics': {k: v for k, v in classification_report(all_true_labels, all_pred_labels, digits=4, output_dict=True).items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
}

metrics_file = os.path.join(args.output, 'evaluation_metrics.json')
with open(metrics_file, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)
logger.info(f"评估指标已保存到: {metrics_file}")

# 生成错误分析
error_indices = np.where(np.array(all_true_labels) != np.array(all_pred_labels))[0]
error_examples = []

for idx in error_indices:
    error_examples.append({
        'true_label': id2label[all_true_labels[idx][0]],
        'predicted_label': id2label[all_pred_labels[idx][0]],
        'count': 1
    })

# 分组相似错误
error_counts = {}
for error in error_examples:
    key = f"{error['true_label']} -> {error['predicted_label']}"
    if key in error_counts:
        error_counts[key]['count'] += 1
    else:
        error_counts[key] = {
            'true_label': error['true_label'],
            'predicted_label': error['predicted_label'],
            'count': 1
        }

# 按频率排序错误
sorted_errors = sorted(error_counts.values(), key=lambda x: x['count'], reverse=True)

# 保存错误分析
error_file = os.path.join(args.output, 'error_analysis.json')
with open(error_file, 'w', encoding='utf-8') as f:
    json.dump(sorted_errors, f, ensure_ascii=False, indent=4)
logger.info(f"错误分析已保存到: {error_file}")

logger.info("评估完成!")
print(f"✅ 评估完成! 结果已保存到 {args.output}")

# 如果是主程序入口点
if __name__ == "__main__":
    print("\n评估完成!")