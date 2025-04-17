# predict_enhanced.py — 增强版预测脚本，支持批量处理和可视化分析

import os
import torch
import argparse
import json
from transformers import BertTokenizerFast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import logging
import time
from model.model_factory import ModelFactory
from utils import get_label_list, clean_entity_name
from config import model_config

# 命令行参数解析
parser = argparse.ArgumentParser(description='中文医疗NER模型预测工具')
parser.add_argument('--input', type=str, help='输入文件路径，每行一个句子，不提供则进入交互模式')
parser.add_argument('--output', type=str, help='输出文件路径，不提供则自动生成')
parser.add_argument('--model', type=str, default=model_config['best_model_path'], help='模型权重路径')
parser.add_argument('--format', type=str, choices=['json', 'text', 'bio'], default='json', help='输出格式：json, text, bio')
parser.add_argument('--batch_size', type=int, default=model_config['eval_batch_size'], help='批处理大小')
parser.add_argument('--pretrained_model', type=str, default=model_config['pretrained_model_name'], 
                    help='预训练模型名称，可选：bert-base-chinese, chinese-medical-bert, pcl-medbert, cmeee-bert, mc-bert, chinese-roberta-med')
parser.add_argument('--use_attention', action='store_true', default=model_config.get('use_attention', False), help='是否使用注意力模型')
parser.add_argument('--use_bilstm', action='store_true', default=model_config.get('use_bilstm', False), help='是否使用BiLSTM层')
parser.add_argument('--no_bilstm', action='store_true', help='禁用BiLSTM层（覆盖默认配置）')
parser.add_argument('--label_path', type=str, default=None, help='标签映射文件路径，如果不提供将使用训练时的标签')
parser.add_argument('--max_length', type=int, default=model_config['max_len'], help='最大序列长度')
parser.add_argument('--pretty', action='store_true', default=True, help='美化JSON输出')
parser.add_argument('--detailed', action='store_true', help='输出详细的实体识别信息，包括位置和类型概率')
args = parser.parse_args()

# no_bilstm 比 use_bilstm 优先级高
use_bilstm = args.use_bilstm
if args.no_bilstm:
    use_bilstm = False

# 生成模型ID和签名，与train_enhanced.py保持一致
model_id = args.pretrained_model.replace('-', '_')
model_type = "attention" if args.use_attention else "base"
bilstm_status = "with_bilstm" if use_bilstm else "no_bilstm"
model_signature = f"{model_id}_{model_type}_{bilstm_status}"

# 设置输出目录
if args.output:
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if not output_dir:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    output_file = args.output
else:
    # 使用时间戳和模型信息生成输出目录和文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', model_id)
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"predict_{model_signature}_{timestamp}.{args.format}")

# 设置日志目录
log_dir = os.path.join(model_config['log_dir'], model_id)
os.makedirs(log_dir, exist_ok=True)

# 设置日志输出
log_file = os.path.join(log_dir, f'prediction_{model_signature}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # 指定编码为utf-8
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 记录预测配置
logger.info("="*50)
logger.info("医学命名实体识别预测开始")
logger.info("="*50)
logger.info(f"预训练模型: {args.pretrained_model}")
logger.info(f"模型类型: {'BERT-Attention-CRF' if args.use_attention else 'BERT-CRF'}")
logger.info(f"BiLSTM层: {'启用' if use_bilstm else '禁用'}")
logger.info(f"模型签名: {model_signature}")
logger.info(f"输出文件: {output_file}")
logger.info(f"输入: {'交互模式' if not args.input else args.input}")
logger.info(f"输出格式: {args.format}")
logger.info(f"批次大小: {args.batch_size}")
logger.info(f"最大序列长度: {args.max_length}")
logger.info(f"详细输出: {args.detailed}")
logger.info("="*50)

# 控制台输出可以保留emoji
print(f"🚀 使用预训练模型: {args.pretrained_model}")
print(f"🔍 模型类型: {'BERT-Attention-CRF' if args.use_attention else 'BERT-CRF'}")
print(f"🧠 BiLSTM层: {'启用' if use_bilstm else '禁用'}")
print(f"📊 模型签名: {model_signature}")
print(f"📁 输出文件: {output_file}")

# 根据参数选择模型路径
model_dir = os.path.join(model_config['model_dir'], model_id)
os.makedirs(model_dir, exist_ok=True)

if args.model == 'best':
    model_path = os.path.join(model_dir, f"best_model_{model_signature}.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, f"best_model_{model_id}.pth")  # 兼容旧版路径
        if not os.path.exists(model_path):
            model_path = model_config['best_model_path']  # 回退到默认路径
elif args.model == 'final':
    model_path = os.path.join(model_dir, f"final_model_{model_signature}.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, f"final_model_{model_id}.pth")  # 兼容旧版路径
        if not os.path.exists(model_path):
            model_path = model_config['final_model_path']  # 回退到默认路径
elif args.model == 'quantized':
    model_path = os.path.join(model_dir, f"quantized_model_{model_signature}.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, f"quantized_model_{model_id}.pth")  # 兼容旧版路径
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "quantized_model.pth")  # 回退到默认路径
elif args.model == 'pruned':
    model_path = os.path.join(model_dir, f"pruned_model_{model_signature}.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, f"pruned_model_{model_id}.pth")  # 兼容旧版路径
        if not os.path.exists(model_path):
            logger.warning(f"警告: 剪枝模型不存在，使用最佳模型代替")
            model_path = os.path.join(model_dir, f"best_model_{model_signature}.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, f"best_model_{model_id}.pth")  # 兼容旧版路径
                if not os.path.exists(model_path):
                    model_path = model_config['best_model_path']  # 再次回退
else:
    # 用户指定的具体路径
    model_path = args.model

logger.info(f"使用模型: {model_path}")
print(f"🔍 使用模型: {model_path}")

# 加载标签列表
if args.label_path and os.path.exists(args.label_path):
    # 从文件加载标签
    logger.info(f"从文件加载标签: {args.label_path}")
    with open(args.label_path, 'r', encoding='utf-8') as f:
        label_list = json.load(f)
else:
    # 使用训练时的标签
    logger.info("使用训练时的标签列表")
    train_paths = [model_config['train_path']]
    if os.path.exists(model_config['dev_path']):
        train_paths.append(model_config['dev_path'])
    if os.path.exists(model_config['test_path']):
        train_paths.append(model_config['test_path'])
    label_list = get_label_list(train_paths)

# 构建标签映射
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label_list)

logger.info(f"标签总数: {num_labels}")
logger.info(f"标签列表: {label_list}")

# 加载分词器和模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")
print(f"💻 使用设备: {device}")

# 使用模型工厂获取与模型匹配的tokenizer
tokenizer = ModelFactory.get_tokenizer_for_model(args.pretrained_model)

# 加载模型 - 使用ModelFactory正确创建模型
model_type_str = 'bert_attention_crf' if args.use_attention else 'bert_crf'
model = ModelFactory.create_model(
    model_type=model_type_str,
    num_labels=num_labels,
    pretrained_model_name=args.pretrained_model
)

# 设置模型的BiLSTM参数（如果适用）
if hasattr(model, 'use_bilstm'):
    model.use_bilstm = use_bilstm
    # 更新BiLSTM相关配置
    if hasattr(model, 'lstm_hidden_size'):
        model.lstm_hidden_size = model_config.get('lstm_hidden_size', 128)
    if hasattr(model, 'lstm_layers'):
        model.lstm_layers = model_config.get('lstm_layers', 1)
    if hasattr(model, 'lstm_dropout'):
        model.lstm_dropout = model_config.get('lstm_dropout', 0.1)

# 加载预训练权重
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"模型加载成功: {model_path}")
    print(f"✅ 模型加载成功: {model_path}")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    print(f"❌ 模型加载失败: {e}")
    print("请确保已经使用相同的预训练模型训练了模型")
    exit(1)

model.to(device)
model.eval()

# 读取输入文件
if args.input and os.path.exists(args.input):
    with open(args.input, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    logger.info(f"从文件加载了 {len(texts)} 条文本")
    print(f"📄 从文件加载了 {len(texts)} 条文本")
else:
    # 交互模式
    if not args.input:
        print("🖋️ 没有提供输入文件，进入交互模式")
        texts = []
        example_text = "患者出现高血压和2型糖尿病，建议服用降压药。"
        print(f"📝 请输入文本进行预测（每行一句，输入空行开始预测，输入'exit'退出）")
        print(f"📝 示例: {example_text}")
        
        while True:
            line = input(">>> ").strip()
            if line.lower() == 'exit':
                if not texts:
                    print("👋 再见!")
                    exit(0)
                else:
                    break
            elif not line and texts:
                break
            elif line:
                texts.append(line)
    else:
        logger.error(f"输入文件不存在: {args.input}")
        print(f"❌ 输入文件不存在: {args.input}")
        exit(1)

# 批量预测函数
def predict_batch(batch_texts):
    encoded_input = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    token_type_ids = encoded_input.get('token_type_ids', None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
    
    # 处理预测结果
    batch_entities = []
    
    for i, text in enumerate(batch_texts):
        # 获取第i个样本的预测标签ID
        if isinstance(outputs, list):
            pred_ids = outputs[i]
        else:
            pred_ids = outputs[i].argmax(dim=-1).tolist()
        
        # 将token映射回原始文本
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        
        entities = []
        entity = None
        orig_tokens = []
        
        # 收集实体
        for j, (token, pred_id) in enumerate(zip(tokens, pred_ids)):
            if token in ['[CLS]', '[SEP]', '[PAD]'] or token.startswith('<'):
                continue
                
            token = token.replace('##', '')
            orig_tokens.append(token)
            
            pred_label = id2label.get(pred_id, 'O')
            
            if pred_label.startswith('B-'):
                if entity:
                    entity_text = ''.join(entity['tokens']).replace('##', '')
                    entity['text'] = entity_text
                    # 清理实体名称
                    entity['text'] = clean_entity_name(entity['text'])
                    entities.append(entity)
                
                entity_type = pred_label[2:]  # 移除"B-"前缀
                entity = {
                    'type': entity_type,
                    'tokens': [token],
                    'start': len(''.join(orig_tokens[:-1])),  # 当前处理的token的位置就是实体开始位置
                    'end': len(''.join(orig_tokens))  # 当前处理token结束的位置
                }
            elif pred_label.startswith('I-') and entity:
                # 确保I-标签类型与当前实体类型一致
                if pred_label[2:] == entity['type']:
                    entity['tokens'].append(token)
                    entity['end'] = len(''.join(orig_tokens))  # 更新实体结束位置
            elif pred_label == 'O':
                if entity:
                    entity_text = ''.join(entity['tokens']).replace('##', '')
                    entity['text'] = entity_text
                    # 清理实体名称
                    entity['text'] = clean_entity_name(entity['text'])
                    entities.append(entity)
                    entity = None
        
        # 处理最后一个实体
        if entity:
            entity_text = ''.join(entity['tokens']).replace('##', '')
            entity['text'] = entity_text
            # 清理实体名称
            entity['text'] = clean_entity_name(entity['text'])
            entities.append(entity)
        
        batch_entities.append({
            'text': text,
            'entities': entities
        })
    
    return batch_entities

# 对所有文本进行预测
all_results = []
total_time = 0
total_entities = 0
start_time = time.time()

for i in range(0, len(texts), args.batch_size):
    batch_texts = texts[i:i+args.batch_size]
    batch_results = predict_batch(batch_texts)
    all_results.extend(batch_results)
    
    # 统计识别到的实体
    for result in batch_results:
        total_entities += len(result['entities'])
    
    # 进度提示
    if len(texts) > args.batch_size:
        print(f"📊 已处理 {min(i+args.batch_size, len(texts))}/{len(texts)} 条文本")

total_time = time.time() - start_time
logger.info(f"预测完成: 处理了 {len(texts)} 条文本, 共识别 {total_entities} 个实体, 耗时 {total_time:.2f} 秒")
print(f"✅ 预测完成: 处理了 {len(texts)} 条文本, 共识别 {total_entities} 个实体, 耗时 {total_time:.2f} 秒")

# 根据输出格式生成结果
if args.format == 'json':
    # 输出JSON格式
    json_results = []
    
    for result in all_results:
        json_result = {'text': result['text'], 'entities': []}
        
        for entity in result['entities']:
            entity_info = {
                'text': entity['text'],
                'type': entity['type'],
                'start': entity['start'],
                'end': entity['end']
            }
            json_result['entities'].append(entity_info)
        
        json_results.append(json_result)
    
    # 保存JSON结果
    with open(output_file, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        else:
            json.dump(json_results, f, ensure_ascii=False)
    
    # 交互模式下输出结果
    if not args.input:
        print("\n📋 预测结果 (JSON格式):")
        for result in json_results:
            print(f"文本: {result['text']}")
            if result['entities']:
                print("识别到的实体:")
                for entity in result['entities']:
                    print(f"  • {entity['type']}: {entity['text']} (位置: {entity['start']}-{entity['end']})")
            else:
                print("没有识别到实体")
            print()

elif args.format == 'text':
    # 输出文本标注格式
    text_results = []
    
    for result in all_results:
        text = result['text']
        entities = sorted(result['entities'], key=lambda x: x['start'])
        
        # 为避免标注位置错乱，先按起始位置排序
        entity_markers = []
        for entity in entities:
            entity_markers.append((entity['start'], f"[{entity['type']}:"))
            entity_markers.append((entity['end'], f"]"))
        
        entity_markers.sort(key=lambda x: (x[0], x[1].endswith("]")))
        
        # 插入标记
        marked_text = ""
        last_pos = 0
        for pos, marker in entity_markers:
            marked_text += text[last_pos:pos] + marker
            last_pos = pos
        
        marked_text += text[last_pos:]
        text_results.append(marked_text)
    
    # 保存文本结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_results))
    
    # 交互模式下输出结果
    if not args.input:
        print("\n📋 预测结果 (文本标注格式):")
        for marked_text in text_results:
            print(marked_text)
            print()

elif args.format == 'bio':
    # 输出BIO格式
    bio_results = []
    
    for result in all_results:
        text = result['text']
        entities = result['entities']
        
        # 初始化所有字符为"O"标签
        bio_tags = ["O"] * len(text)
        
        # 插入实体标签
        for entity in entities:
            entity_type = entity['type']
            start = entity['start']
            end = entity['end']
            
            # 设置B-标签（实体的第一个字符）
            if start < len(bio_tags):
                bio_tags[start] = f"B-{entity_type}"
            
            # 设置I-标签（实体的其他字符）
            for i in range(start + 1, end):
                if i < len(bio_tags):
                    bio_tags[i] = f"I-{entity_type}"
        
        # 拼接字符和标签
        bio_lines = []
        for char, tag in zip(text, bio_tags):
            bio_lines.append(f"{char}\t{tag}")
        
        bio_results.append('\n'.join(bio_lines))
    
    # 保存BIO结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(bio_results))
    
    # 交互模式下输出结果
    if not args.input:
        print("\n📋 预测结果 (BIO格式):")
        for bio_text in bio_results[:3]:  # 限制输出前几个结果，避免太长
            print(bio_text)
            print()
        if len(bio_results) > 3:
            print("... （更多结果已保存到文件）")
    
logger.info(f"结果已保存至: {output_file}")
print(f"📁 结果已保存至: {output_file}")

# 当在交互模式下时，提供一些统计信息
if not args.input:
    # 统计实体类型分布
    entity_counts = {}
    for result in all_results:
        for entity in result['entities']:
            entity_type = entity['type']
            if entity_type not in entity_counts:
                entity_counts[entity_type] = 0
            entity_counts[entity_type] += 1
    
    if entity_counts:
        print("\n📊 实体类型分布:")
        for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {entity_type}: {count} 个")
    else:
        print("\n❗ 未识别到任何实体")