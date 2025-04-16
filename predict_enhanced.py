# predict_enhanced.py — 增强版预测脚本，支持批量处理和可视化分析

import os
import torch
import argparse
import json
from transformers import BertTokenizerFast
from model.bert_crf_model import BertCRF
from model.bert_attention_crf_model import BertAttentionCRF
from utils import get_label_list
from model_explainability import ModelExplainer
from config import model_config, explainability_config
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 命令行参数解析
parser = argparse.ArgumentParser(description='医学命名实体识别预测工具')
parser.add_argument('--input', type=str, help='输入文本或文件路径')
parser.add_argument('--batch', action='store_true', help='是否批量处理文件')
parser.add_argument('--model', type=str, default='best', choices=['best', 'final', 'quantized'], help='使用的模型类型')
parser.add_argument('--output', type=str, default='results/predictions', help='输出目录')
parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
parser.add_argument('--attention', action='store_true', help='是否使用注意力模型')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output, exist_ok=True)

# 加载模型和标签信息
bert_model_name = model_config['bert_model_name']
data_files = [model_config['train_path'], model_config['dev_path'], model_config['test_path']]

# 根据参数选择模型路径
if args.model == 'best':
    model_path = model_config['best_model_path']
elif args.model == 'final':
    model_path = model_config['final_model_path']
elif args.model == 'quantized':
    model_path = os.path.join(model_config['model_dir'], "quantized_model.pth")
    if not os.path.exists(model_path):
        print(f"警告: 量化模型不存在，使用最佳模型代替")
        model_path = model_config['best_model_path']

# 构建标签映射
label_list = get_label_list(data_files)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label_list)

# 加载分词器和模型
tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 根据参数选择模型类型
if args.attention:
    print(f"使用BERT-Attention-CRF模型: {model_path}")
    model = BertAttentionCRF(bert_model_name, num_labels)
else:
    print(f"使用BERT-CRF模型: {model_path}")
    model = BertCRF(bert_model_name, num_labels)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 初始化模型解释器
explainer = ModelExplainer(model, tokenizer, id2label, device)

# 预测单个文本
def predict_text(text):
    # 将句子切成字列表
    tokens = list(text.strip())
    encoding = tokenizer(tokens,
                         is_split_into_words=True,
                         return_tensors="pt",
                         padding="max_length",
                         truncation=True,
                         max_length=128,
                         return_offsets_mapping=True)

    encoding = {k: v.to(device) for k, v in encoding.items() if k != 'offset_mapping'}

    with torch.no_grad():
        predictions = model(encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids'])

    # 解码标签，只保留非特殊符号的部分（剔除[CLS], [SEP], padding）
    word_ids = encoding['input_ids'][0].tolist()
    tokens_decoded = tokenizer.convert_ids_to_tokens(word_ids)
    preds = predictions[0]  # batch size = 1

    results = []
    for token, pred_label in zip(tokens_decoded, preds):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        label = id2label[pred_label]
        results.append((token, label))
    
    return results[:len(tokens)]  # 确保结果长度与输入一致

# 生成实体统计信息
def generate_entity_statistics(all_predictions):
    entity_counts = {}
    entity_examples = {}
    
    for text, predictions in all_predictions:
        # 提取实体
        current_entity = []
        current_type = None
        
        for token, label in predictions:
            if label.startswith('B-'):
                # 如果有正在处理的实体，保存它
                if current_entity:
                    entity = ''.join(current_entity)
                    if current_type not in entity_counts:
                        entity_counts[current_type] = 0
                        entity_examples[current_type] = []
                    entity_counts[current_type] += 1
                    if len(entity_examples[current_type]) < 5:  # 每种类型最多保存5个例子
                        entity_examples[current_type].append(entity)
                
                # 开始新实体
                current_entity = [token]
                current_type = label[2:]
            elif label.startswith('I-') and current_entity:
                # 继续当前实体
                current_entity.append(token)
            else:
                # 非实体标签，如果有正在处理的实体，保存它
                if current_entity:
                    entity = ''.join(current_entity)
                    if current_type not in entity_counts:
                        entity_counts[current_type] = 0
                        entity_examples[current_type] = []
                    entity_counts[current_type] += 1
                    if len(entity_examples[current_type]) < 5:  # 每种类型最多保存5个例子
                        entity_examples[current_type].append(entity)
                    current_entity = []
                    current_type = None
        
        # 处理最后一个实体
        if current_entity:
            entity = ''.join(current_entity)
            if current_type not in entity_counts:
                entity_counts[current_type] = 0
                entity_examples[current_type] = []
            entity_counts[current_type] += 1
            if len(entity_examples[current_type]) < 5:
                entity_examples[current_type].append(entity)
    
    return entity_counts, entity_examples

# 可视化实体统计
def visualize_entity_statistics(entity_counts, output_path):
    if not entity_counts:
        return
    
    # 创建条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(entity_counts.keys()), y=list(entity_counts.values()))
    plt.title('实体类型分布')
    plt.xlabel('实体类型')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 批量处理文件
def process_file(file_path, output_dir):
    print(f"\n处理文件: {file_path}")
    all_predictions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行
    for i, line in enumerate(tqdm(lines, desc="预测进度")):
        line = line.strip()
        if not line:
            continue
        
        # 预测
        predictions = predict_text(line)
        all_predictions.append((line, predictions))
    
    # 保存预测结果
    output_file = os.path.join(output_dir, "predictions.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text, preds in all_predictions:
            f.write(f"文本: {text}\n")
            f.write("标注结果:\n")
            for token, label in preds:
                f.write(f"{token}\t{label}\n")
            f.write("\n")
    
    # 生成实体统计
    entity_counts, entity_examples = generate_entity_statistics(all_predictions)
    
    # 保存统计结果
    stats = {
        "entity_counts": entity_counts,
        "entity_examples": entity_examples,
        "total_texts": len(all_predictions),
        "total_entities": sum(entity_counts.values())
    }
    
    with open(os.path.join(output_dir, "statistics.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 可视化统计结果
    visualize_entity_statistics(entity_counts, os.path.join(output_dir, "entity_distribution.png"))
    
    # 生成HTML报告
    generate_html_report(all_predictions, stats, os.path.join(output_dir, "report.html"))
    
    print(f"\n处理完成，结果已保存至 {output_dir}")
    print(f"共处理 {len(all_predictions)} 条文本，识别出 {sum(entity_counts.values())} 个实体")
    print("实体分布:")
    for entity_type, count in entity_counts.items():
        print(f"  - {entity_type}: {count}")

# 生成HTML报告
def generate_html_report(all_predictions, stats, output_path):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>医学命名实体识别报告</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1000px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .entity { display: inline-block; padding: 2px 5px; border-radius: 3px; margin: 2px; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            .examples { margin-top: 10px; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>医学命名实体识别报告</h1>
            
            <div class="summary">
                <h2>统计摘要</h2>
                <p>总文本数: {total_texts}</p>
                <p>总实体数: {total_entities}</p>
                
                <h3>实体类型分布</h3>
                <table>
                    <tr>
                        <th>实体类型</th>
                        <th>数量</th>
                        <th>示例</th>
                    </tr>
    """.format(total_texts=stats["total_texts"], total_entities=stats["total_entities"])
    
    # 添加实体统计
    for entity_type, count in stats["entity_counts"].items():
        examples = stats["entity_examples"].get(entity_type, [])
        examples_str = ", ".join(examples)
        html_content += f"""
                    <tr>
                        <td>{entity_type}</td>
                        <td>{count}</td>
                        <td>{examples_str}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <h2>预测结果</h2>
            <table>
                <tr>
                    <th>序号</th>
                    <th>原文本</th>
                    <th>标注结果</th>
                </tr>
    """
    
    # 添加预测结果
    for i, (text, predictions) in enumerate(all_predictions):
        # 生成带标记的文本
        marked_text = ""
        current_entity = []
        current_type = None
        entity_colors = {}
        
        for token, label in predictions:
            if label.startswith('B-'):
                # 如果有正在处理的实体，关闭它
                if current_entity:
                    entity_text = ''.join(current_entity)
                    color = entity_colors.get(current_type, "#" + hex(hash(current_type) % 0xffffff)[2:].zfill(6))
                    marked_text += f"</span>"
                
                # 开始新实体
                current_type = label[2:]
                if current_type not in entity_colors:
                    # 为每种实体类型分配一个颜色
                    hue = hash(current_type) % 360
                    entity_colors[current_type] = f"hsl({hue}, 70%, 80%)"
                
                color = entity_colors[current_type]
                marked_text += f"<span class='entity' style='background-color: {color};' title='{current_type}'>"
                marked_text += token
                current_entity = [token]
            elif label.startswith('I-'):
                # 继续当前实体
                marked_text += token
                current_entity.append(token)
            else:
                # 非实体标签，如果有正在处理的实体，关闭它
                if current_entity:
                    marked_text += f"</span>"
                    current_entity = []
                    current_type = None
                
                marked_text += token
        
        # 处理最后一个实体
        if current_entity:
            marked_text += f"</span>"
        
        # 添加到HTML
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{text}</td>
                    <td>{marked_text}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

# 交互式预测
def interactive_predict():
    print("\n=== 医学命名实体识别交互式预测 ===\n")
    print(f"使用模型: {model_path}")
    print("支持的实体类型:", ", ".join([label for label in label_list if label != "O"]))
    print("输入 'exit' 退出\n")
    
    while True:
        text = input("请输入医学文本: ")
        if text.lower() in ["exit", "quit", "q"]:
            break
        
        if not text.strip():
            continue
        
        # 预测
        predictions = predict_text(text)
        
        # 打印结果
        print("\n标注结果:")
        for token, label in predictions:
            print(f"{token}\t{label}")
        
        # 提取实体
        entities = []
        current_entity = []
        current_type = None
        
        for token, label in predictions:
            if label.startswith('B-'):
                # 如果有正在处理的实体，保存它
                if current_entity:
                    entities.append((current_type, ''.join(current_entity)))
                
                # 开始新实体
                current_entity = [token]
                current_type = label[2:]
            elif label.startswith('I-') and current_entity:
                # 继续当前实体
                current_entity.append(token)
            else:
                # 非实体标签，如果有正在处理的实体，保存它
                if current_entity:
                    entities.append((current_type, ''.join(current_entity)))
                    current_entity = []
                    current_type = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append((current_type, ''.join(current_entity)))
        
        # 打印提取的实体
        if entities:
            print("\n提取的实体:")
            for entity_type, entity_text in entities:
                print(f"  - {entity_type}: {entity_text}")
        else:
            print("\n未提取到实体")
        
        # 可视化（如果启用）
        if args.visualize:
            vis_dir = os.path.join(args.output, "interactive")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 生成可视化
            try:
                explainer.visualize_prediction_confidence(text, os.path.join(vis_dir, "prediction_confidence.png"))
                explainer.generate_html_visualization(text, os.path.join(vis_dir, "visualization.html"))
                print(f"\n可视化结果已保存至 {vis_dir}")
            except Exception as e:
                print(f"\n生成可视化失败: {e}")
        
        print("\n" + "-"*50)

# 主函数
def main():
    if args.batch and args.input:
        # 批量处理文件
        if not os.path.exists(args.input):
            print(f"错误: 文件 {args.input} 不存在")
            return
        
        process_file(args.input, args.output)
    elif args.input:
        # 处理单个文本
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        predictions = predict_text(text)
        
        # 打印结果
        print("\n标注结果:")
        for token, label in predictions:
            print(f"{token}\t{label}")
        
        # 可视化（如果启用）
        if args.visualize:
            vis_dir = os.path.join(args.output, "single")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 生成可视化
            try:
                explainer.visualize_prediction_confidence(text, os.path.join(vis_dir, "prediction_confidence.png"))
                explainer.generate_html_visualization(text, os.path.join(vis_dir, "visualization.html"))
                print(f"\n可视化结果已保存至 {vis_dir}")
            except Exception as e:
                print(f"\n生成可视化失败: {e}")
    else:
        # 交互式预测
        interactive_predict()

if __name__ == "__main__":
    main()