# predict_enhanced.py — 增强版预测脚本，支持批量处理和可视化分析

import os
import torch
import argparse
import json
from transformers import BertTokenizerFast
from model.model_factory import ModelFactory
from utils import get_label_list
from model_explainability import ModelExplainer
from config import model_config, explainability_config
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import time
from model.bert_crf_model import BertCRF
from model.bert_attention_crf_model import BertAttentionCRF

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
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 记录预测配置
logger.info("="*50)
logger.info("医学命名实体识别预测开始")
logger.info("="*50)
logger.info(f"🚀 预训练模型: {args.pretrained_model}")
logger.info(f"🔍 模型类型: {'BERT-Attention-CRF' if args.use_attention else 'BERT-CRF'}")
logger.info(f"🧠 BiLSTM层: {'启用' if use_bilstm else '禁用'}")
logger.info(f"📊 模型签名: {model_signature}")
logger.info(f"📁 输出文件: {output_file}")
logger.info(f"输入: {'交互模式' if not args.input else args.input}")
logger.info(f"输出格式: {args.format}")
logger.info(f"批次大小: {args.batch_size}")
logger.info(f"最大序列长度: {args.max_length}")
logger.info(f"详细输出: {args.detailed}")
logger.info("="*50)

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

# 使用模型工厂创建模型和获取分词器
tokenizer = ModelFactory.get_tokenizer_for_model(args.pretrained_model)
  
# 根据参数选择模型类型
if args.use_attention:
    model = BertAttentionCRF.from_pretrained(
        args.pretrained_model,
        num_labels=num_labels,
        use_bilstm=use_bilstm,
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
        use_bilstm=use_bilstm,
        lstm_hidden_size=model_config['lstm_hidden_size'],
        lstm_layers=model_config['lstm_layers'],
        lstm_dropout=model_config['lstm_dropout']
    )

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

# 初始化模型解释器
explainer = ModelExplainer(model, tokenizer, id2label, device)

# 读取输入文件
if args.input and os.path.exists(args.input):
    with open(args.input, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    logger.info(f"从文件加载了 {len(texts)} 个文本样本")
    print(f"📄 从文件加载了 {len(texts)} 个文本样本")
elif args.input and not os.path.exists(args.input):
    logger.error(f"输入文件不存在: {args.input}")
    print(f"❌ 输入文件不存在: {args.input}")
    texts = [args.input]  # 将输入参数作为单个文本样本
    logger.info("将命令行参数作为单个文本样本处理")
    print("🔤 将命令行参数作为单个文本样本处理")
else:
    # 交互模式
    logger.info("进入交互模式")
    print("📝 请输入文本进行命名实体识别（输入'quit'或'exit'退出）：")
    texts = []
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        texts.append(user_input)
    
    if not texts:
        logger.info("未输入任何文本，退出")
        print("❌ 未输入任何文本，退出")
        exit(0)
    
    logger.info(f"交互模式收集了 {len(texts)} 个文本样本")

# 预测函数
def predict_batch(batch_texts):
    encoded_inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)
    token_type_ids = encoded_inputs.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    
    with torch.no_grad():
        if token_type_ids is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
    # 处理不同模型输出格式  
    if isinstance(outputs, list):
        # CRF模型直接返回标签序列
        predicted_label_ids = outputs
    else:
        # 非CRF模型，需要从logits中获取标签
        predicted_label_ids = outputs.argmax(dim=2).cpu().numpy()
    
    # 处理结果
    batch_results = []
    for i, text in enumerate(batch_texts):
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        
        # 根据模型输出类型获取预测的标签ID
        if isinstance(outputs, list):
            pred_label_ids = outputs[i] 
        else:
            pred_label_ids = predicted_label_ids[i]
        
        # 处理每个token
        result_labels = []
        entities = []
        current_entity = None
        
        for j, (token, pred_id) in enumerate(zip(text_tokens, pred_label_ids)):
            if token in ["[CLS]", "[SEP]", "[PAD]"] or token.startswith("<"):
                # 如果当前有正在处理的实体，结束它
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue
                
            # 恢复原始文本 (处理WordPiece分词)
            token_text = token.replace("##", "")
            
            # 获取预测标签
            label = id2label[pred_id]
            result_labels.append(label)
            
            # 处理实体
            if label.startswith("B-"):
                # 如果当前有正在处理的实体，结束它
                if current_entity is not None:
                    entities.append(current_entity)
                
                # 开始一个新实体
                entity_type = label[2:]  # 去掉"B-"前缀
                current_entity = {
                    "type": entity_type,
                    "text": token_text,
                    "start": j-1 if j > 0 else 0,  # 近似位置，需要后处理校正
                    "end": j
                }
            elif label.startswith("I-") and current_entity is not None:
                # 继续当前实体
                entity_type = label[2:]  # 去掉"I-"前缀
                if entity_type == current_entity["type"]:
                    current_entity["text"] += token_text
                    current_entity["end"] = j
            elif current_entity is not None:
                # 结束当前实体
                entities.append(current_entity)
                current_entity = None
        
        # 处理最后一个实体（如果有）
        if current_entity is not None:
            entities.append(current_entity)
        
        # 校正实体位置
        corrected_entities = []
        for entity in entities:
            # 在原始文本中查找实体文本，找到实际位置
            entity_text = entity["text"]
            entity_type = entity["type"]
            
            # 如果实体文本在原始文本中能找到，使用实际位置
            start_pos = text.find(entity_text)
            if start_pos != -1:
                corrected_entities.append({
                    "type": entity_type,
                    "text": entity_text,
                    "start": start_pos,
                    "end": start_pos + len(entity_text)
                })
            else:
                # 否则使用近似位置
                corrected_entities.append(entity)
        
        # 添加到批次结果
        batch_results.append({
            "text": text,
            "labels": result_labels,
            "entities": corrected_entities
        })
    
    return batch_results

# 分批处理文本
results = []
batch_size = args.batch_size
num_batches = (len(texts) + batch_size - 1) // batch_size  # 向上取整

start_time = time.time()
for i in tqdm(range(num_batches), desc="预测进度"):
    start_idx = i * batch_size
    end_idx = min(start_idx + batch_size, len(texts))
    batch_texts = texts[start_idx:end_idx]
    
    batch_results = predict_batch(batch_texts)
    results.extend(batch_results)

end_time = time.time()
processing_time = end_time - start_time
avg_time_per_sample = processing_time / len(texts)

logger.info(f"预测完成! 处理时间: {processing_time:.2f}秒, 每样本平均: {avg_time_per_sample:.4f}秒")
print(f"✅ 预测完成! 处理 {len(texts)} 个样本用时 {processing_time:.2f}秒, 平均每样本 {avg_time_per_sample:.4f}秒")

# 根据输出格式保存结果
if args.format == 'json':
    # JSON格式输出 - 包含完整的实体信息
    output_data = []
    for result in results:
        # 如果不需要详细信息，只保留必要的字段
        if not args.detailed:
            output_data.append({
                "text": result["text"],
                "entities": [
                    {"type": e["type"], "text": e["text"]} 
                    for e in result["entities"]
                ]
            })
        else:
            output_data.append(result)
    
    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(output_data, f, ensure_ascii=False)
            
elif args.format == 'text':
    # 文本格式输出 - 每个样本一行，带有标注的实体
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            text = result["text"]
            entities = result["entities"]
            
            # 按照结束位置从大到小排序，以便从后向前处理文本
            entities = sorted(entities, key=lambda e: e["end"], reverse=True)
            
            # 在文本中标记实体
            marked_text = text
            for entity in entities:
                start = entity["start"]
                end = entity["end"]
                entity_type = entity["type"]
                
                # 使用特殊标记突出显示实体
                marked_text = (
                    marked_text[:start] + 
                    f"[{marked_text[start:end]}:{entity_type}]" + 
                    marked_text[end:]
                )
            
            f.write(f"{marked_text}\n")
            
elif args.format == 'bio':
    # BIO格式输出 - 每行一个字符和对应的BIO标签
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            text = result["text"]
            entities = result["entities"]
            
            # 创建默认标签(全部为O)
            bio_tags = ["O"] * len(text)
            
            # 为所有实体分配BIO标签
            for entity in entities:
                start = entity["start"]
                end = entity["end"]
                entity_type = entity["type"]
                
                # 分配B标签给实体的第一个字符
                bio_tags[start] = f"B-{entity_type}"
                
                # 分配I标签给实体的剩余字符
                for i in range(start + 1, end):
                    bio_tags[i] = f"I-{entity_type}"
            
            # 输出字符和对应的BIO标签
            for char, tag in zip(text, bio_tags):
                f.write(f"{char} {tag}\n")
            f.write("\n")  # 不同样本之间的空行

logger.info(f"预测结果已保存到: {output_file}")
print(f"✅ 预测结果已保存到: {output_file}")

# 可视化结果统计（如果样本数超过1）
if len(results) > 1:
    # 统计实体类型分布
    entity_types = {}
    for result in results:
        for entity in result["entities"]:
            entity_type = entity["type"]
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
    
    # 生成统计图表
    if entity_types:
        plt.figure(figsize=(10, 6))
        plt.bar(entity_types.keys(), entity_types.values())
        plt.title('实体类型分布')
        plt.xlabel('实体类型')
        plt.ylabel('数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.splitext(output_file)[0] + "_entity_distribution.png"
        plt.savefig(chart_path)
        logger.info(f"实体类型分布图已保存到: {chart_path}")
        print(f"📊 实体类型分布图已保存到: {chart_path}")