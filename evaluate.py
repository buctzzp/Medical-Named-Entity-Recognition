# evaluate.py 增强版评估脚本，支持命令行参数、混淆矩阵和批量评估

import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from model.bert_crf_model import BertCRF
from utils import NERDataset, get_label_list
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from config import model_config

# 命令行参数解析
parser = argparse.ArgumentParser(description='医学命名实体识别评估工具')
parser.add_argument('--data', type=str, default=model_config['test_path'], help='评估数据集路径')
parser.add_argument('--model', type=str, default=model_config['final_model_path'], help='模型路径')
parser.add_argument('--batch_size', type=int, default=model_config['eval_batch_size'], help='评估批次大小')
parser.add_argument('--output', type=str, default='results', help='输出目录')
parser.add_argument('--confusion_matrix', action='store_true', help='是否生成混淆矩阵')
args = parser.parse_args()

# 加载模型和标签映射
tokenizer = BertTokenizerFast.from_pretrained(model_config['bert_model_name'])
label_list = get_label_list([
    model_config['train_path'], 
    model_config['dev_path'], 
    model_config['test_path']
])
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型
model = BertCRF(model_config['bert_model_name'], len(label_list))
model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
model.to(device)
model.eval()

# 加载测试数据
print(f"评估数据集: {args.data}")
test_dataset = NERDataset(args.data, tokenizer, label2id, model_config['max_len'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

true_labels = []
predictions = []
all_true_labels_flat = []
all_pred_labels_flat = []
all_char_preds = []  # 收集所有预测（用于写入简洁版本）

# 创建输出目录
os.makedirs(args.output, exist_ok=True)
verbose_output_path = os.path.join(args.output, "test_predictions_verbose.txt")
simple_output_path = os.path.join(args.output, "test_predictions.txt")

# 打开详细输出文件
with open(verbose_output_path, "w", encoding="utf-8") as fout:
    # 格式化输出表头
    fout.write("{:<6}{:<8}{:<15}{:<15}{}\n".format("位置", "字符", "真实标签", "预测标签", "是否正确"))
    fout.write("-" * 60 + "\n")
    
    # 使用tqdm显示进度条
    for batch in tqdm(test_loader, desc="评估进度"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            pred_ids_batch = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
        
        # 处理批次中的每个样本
        for i in range(batch['input_ids'].size(0)):
            input_ids = batch['input_ids'][i]
            label_ids = batch['labels'][i].tolist()
            pred_ids = pred_ids_batch[i]
            
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            true_seq = []
            pred_seq = []
            char_seq = []
            
            for token, true_id, pred_id in zip(tokens, label_ids, pred_ids):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                if true_id == -100:
                    continue
                    
                true_label = id2label[true_id]
                pred_label = id2label[pred_id]
                token_char = token.replace("##", "")
                
                true_seq.append(true_label)
                pred_seq.append(pred_label)
                char_seq.append(token_char)
                
                # 收集用于混淆矩阵的标签
                all_true_labels_flat.append(true_label)
                all_pred_labels_flat.append(pred_label)
            
            if len(true_seq) > 0:
                true_labels.append(true_seq)
                predictions.append(pred_seq)
                all_char_preds.append(list(zip(char_seq, pred_seq)))
                
                # 写入详细预测结果
                for j, (ch, tl, pl) in enumerate(zip(char_seq, true_seq, pred_seq)):
                    mark = "✓" if tl == pl else "✗"
                    fout.write("{:<6}{:<8}{:<15}{:<15}{}\n".format(j, ch, tl, pl, mark))
                fout.write("\n")

# 计算并打印评估指标
print("\n================== 分类指标报告 ==================")
report = classification_report(true_labels, predictions, digits=4)
print(report)

# 计算总体指标
p = precision_score(true_labels, predictions)
r = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
print(f"\n总体指标 - 精确率: {p:.4f}, 召回率: {r:.4f}, F1分数: {f1:.4f}")

# 保存分类报告到文件
report_path = os.path.join(args.output, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
    f.write(f"\n总体指标 - 精确率: {p:.4f}, 召回率: {r:.4f}, F1分数: {f1:.4f}")

# 保存精简预测结果（完整句子+标签）
with open(simple_output_path, "w", encoding="utf-8") as f:
    for char_label_pair in all_char_preds:
        for ch, label in char_label_pair:
            f.write(f"{ch}\t{label}\n")
        f.write("\n")

# 生成混淆矩阵（如果需要）
if args.confusion_matrix and len(set(all_true_labels_flat)) > 1:
    # 获取唯一标签（按字母顺序排序）
    unique_labels = sorted(set(all_true_labels_flat))
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_true_labels_flat, all_pred_labels_flat, labels=unique_labels)
    
    # 创建DataFrame以便于可视化
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    
    # 绘制混淆矩阵热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("NER标签混淆矩阵")
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    
    # 保存混淆矩阵图
    cm_path = os.path.join(args.output, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"混淆矩阵已保存至 {cm_path}")

# 打印输出文件路径
print(f"\n预测结果已保存至 {simple_output_path}")
print(f"详细对齐分析结果已保存至 {verbose_output_path}")
print(f"分类报告已保存至 {report_path}")

# 添加错误分析统计
error_count = sum(1 for true, pred in zip(all_true_labels_flat, all_pred_labels_flat) if true != pred)
total_count = len(all_true_labels_flat)
error_rate = error_count / total_count if total_count > 0 else 0

print(f"\n错误分析统计:")
print(f"总标签数: {total_count}")
print(f"错误预测数: {error_count}")
print(f"错误率: {error_rate:.4f} ({error_count}/{total_count})")

# 如果是主程序入口点
if __name__ == "__main__":
    print("\n评估完成!")