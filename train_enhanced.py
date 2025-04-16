# train_enhanced.py - 增强版训练脚本，整合所有优化功能

import os
import torch
import logging
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_scheduler
from model.bert_crf_model import BertCRF
from model.bert_attention_crf_model import BertAttentionCRF
from utils import NERDataset, get_label_list
from data_augmentation import DataAugmentation, create_sample_dictionaries
from model_optimization import ModelOptimizer
from model_explainability import ModelExplainer
import matplotlib.pyplot as plt
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from config import model_config, augmentation_config, optimization_config, explainability_config
from datetime import datetime
import json

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 检测GPU
print("✅ 是否检测到 GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🖥️ 当前 GPU 名称:", torch.cuda.get_device_name(0))
    print("🔥 当前设备:", torch.device("cuda"))
else:
    print("❌ 当前使用的是 CPU")

# 设置日志
log_dir = model_config['log_dir']
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs(model_config['model_dir'], exist_ok=True)
os.makedirs(explainability_config['visualization_output_dir'], exist_ok=True)

# 加载分词器
tokenizer = BertTokenizerFast.from_pretrained(model_config['bert_model_name'])

# 构建标签映射
label_list = get_label_list([
    model_config['train_path'],
    model_config['dev_path'],
    model_config['test_path']
])
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label_list)

logger.info(f"标签总数: {num_labels}")
logger.info(f"标签列表: {label_list}")

# 数据增强
if augmentation_config['use_data_augmentation']:
    logger.info("正在准备数据增强...")
    
    # 检查同义词和实体词典是否存在，不存在则创建示例词典
    if not os.path.exists(augmentation_config['synonym_dict_path']) or \
       not os.path.exists(augmentation_config['entity_dict_path']):
        logger.info("未找到词典文件，创建示例词典...")
        create_sample_dictionaries()
    
    # 初始化数据增强器
    augmenter = DataAugmentation(
        synonym_dict_path=augmentation_config['synonym_dict_path'],
        entity_dict_path=augmentation_config['entity_dict_path']
    )
    
    # 读取原始训练数据
    with open(model_config['train_path'], 'r', encoding='utf-8') as f:
        train_data = f.read().strip().split('\n\n')
    
    # 增强数据
    augmented_data = []
    for sample in tqdm(train_data, desc="数据增强"):
        if not sample.strip():
            continue
        
        # 解析样本
        lines = sample.strip().split('\n')
        words, labels = [], []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 2:
                    words.append(parts[0])
                    labels.append(parts[1])
        
        # 应用数据增强
        if random.random() < 0.7:  # 只对70%的样本进行增强
            aug_words, aug_labels = augmenter.augment_data(
                words, labels,
                use_synonym_replace=augmentation_config['use_synonym_replace'],
                use_entity_replace=augmentation_config['use_entity_replace'],
                synonym_replace_ratio=augmentation_config['synonym_replace_ratio'],
                entity_replace_ratio=augmentation_config['entity_replace_ratio']
            )
            
            # 添加增强后的样本
            aug_sample = '\n'.join([f"{w}\t{l}" for w, l in zip(aug_words, aug_labels)])
            augmented_data.append(aug_sample)
    
    # 合并原始数据和增强数据
    all_data = train_data + augmented_data
    logger.info(f"数据增强完成: 原始样本数 {len(train_data)}, 增强后样本数 {len(all_data)}")
    
    # 保存增强后的数据
    augmented_train_path = model_config['train_path'].replace('.txt', '_augmented.txt')
    with open(augmented_train_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_data))
    
    # 更新训练数据路径
    train_path = augmented_train_path
else:
    train_path = model_config['train_path']

# 加载数据集
train_dataset = NERDataset(train_path, tokenizer, label2id, model_config['max_len'])
dev_dataset = NERDataset(model_config['dev_path'], tokenizer, label2id, model_config['max_len'])

train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=model_config['eval_batch_size'])

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 选择模型类型
if model_config['use_self_attention']:
    logger.info("使用增强版BERT-Attention-CRF模型")
    model = BertAttentionCRF(model_config['bert_model_name'], num_labels).to(device)
else:
    logger.info("使用标准BERT-CRF模型")
    model = BertCRF(model_config['bert_model_name'], num_labels).to(device)

# 打印模型结构信息
logger.info("模型结构配置:")
logger.info(f"✓ BiLSTM层状态: {'启用' if model_config['use_bilstm'] else '禁用'}")
if model_config['use_bilstm']:
    logger.info(f"  - 隐藏层大小: {model_config['lstm_hidden_size']}")
    logger.info(f"  - LSTM层数: {model_config['lstm_layers']}")
    logger.info(f"  - Dropout率: {model_config['lstm_dropout']}")

# 初始化模型优化器
model_optimizer = ModelOptimizer(model, device)

# 优化器配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=model_config['learning_rate'],
    weight_decay=model_config['weight_decay']
)

# 学习率调度器（带预热）
num_training_steps = len(train_loader) * model_config['num_epochs']
num_warmup_steps = int(num_training_steps * model_config['warmup_ratio'])
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 验证集评估函数
def evaluate_on_dev():
    model.eval()
    true_labels = []
    pred_labels = []
    eval_loss = 0
    num_batches = 0
    start_time = datetime.now()
    
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            loss = model.crf.neg_log_likelihood(
                pred,
                batch['labels'],
                batch['attention_mask']
            )
            eval_loss += loss.item()
            num_batches += 1
            
            # 处理批次中的每个样本
            for i in range(batch['input_ids'].size(0)):
                label_ids = batch['labels'][i].tolist()
                pred_ids = pred[i]
                
                tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                word_labels = []
                word_preds = []
                
                for token, true_id, pred_id in zip(tokens, label_ids, pred_ids):
                    if token in ["[CLS]", "[SEP]", "[PAD]"] or true_id == -100:
                        continue
                    word_labels.append(id2label[true_id])
                    word_preds.append(id2label[pred_id])
                
                if word_labels:
                    true_labels.append(word_labels)
                    pred_labels.append(word_preds)
    
    # 计算评估指标
    eval_loss = eval_loss / num_batches
    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    # 获取每个标签的详细评估报告
    report = classification_report(true_labels, pred_labels, output_dict=True)
    
    # 计算评估耗时
    eval_time = (datetime.now() - start_time).total_seconds()
    
    return {
        'precision': p,
        'recall': r,
        'f1': f1,
        'loss': eval_loss,
        'report': report,
        'eval_time': eval_time,
        'true_labels': true_labels,
        'pred_labels': pred_labels
    }

# 训练循环

# 记录训练过程中的指标
train_history = {
    'train_losses': [],
    'eval_losses': [],
    'precisions': [],
    'recalls': [],
    'f1_scores': [],
    'learning_rates': []
}
best_f1 = 0
early_stop_counter = 0
patience = model_config['early_stopping_patience']

for epoch in range(model_config['num_epochs']):
    model.train()
    total_loss = 0
    epoch_start_time = datetime.now()
    
    # 使用tqdm显示训练进度
    progress_bar = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{model_config['num_epochs']}")
    
    for step, batch in enumerate(progress_bar):
        # 使用混合精度训练
        if model_config['use_amp']:
            loss = model_optimizer.train_step_amp(batch, optimizer)
        else:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['labels'])
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['max_grad_norm'])
            
            optimizer.step()
            optimizer.zero_grad()
        
        lr_scheduler.step()
        total_loss += loss
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        
        # 定期记录训练信息
        if (step + 1) % model_config['logging_steps'] == 0:
            logger.info(
                f'Epoch: {epoch+1}/{model_config["num_epochs"]}, '
                f'Step: {step+1}/{len(train_loader)}, '
                f'Loss: {loss:.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )
    
    # 应用模型剪枝（如果启用）
    if optimization_config['use_model_pruning'] and epoch == optimization_config['apply_pruning_epoch']:
        logger.info(f"在第 {epoch+1} 个epoch应用模型剪枝..")
        model_optimizer.apply_pruning(
            amount=optimization_config['pruning_amount'],
            method=optimization_config['pruning_method']
        )
        logger.info("模型剪枝完成")
    
    # 计算平均损失
    avg_train_loss = total_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # 在验证集上评估
    eval_results = evaluate_on_dev()
    
    # 更新训练历史
    train_history['train_losses'].append(avg_train_loss)
    train_history['eval_losses'].append(eval_results['loss'])
    train_history['precisions'].append(eval_results['precision'])
    train_history['recalls'].append(eval_results['recall'])
    train_history['f1_scores'].append(eval_results['f1'])
    train_history['learning_rates'].append(current_lr)
    
    # 计算训练耗时
    epoch_time = (datetime.now() - epoch_start_time).total_seconds()
    
    # 输出详细的评估报告
    logger.info(f"\n{'='*50}\nEpoch {epoch + 1} 训练报告\n{'='*50}")
    logger.info(f"训练时间: {epoch_time:.2f}秒")
    logger.info(f"评估时间: {eval_results['eval_time']:.2f}秒")
    logger.info(f"\n训练指标:")
    logger.info(f"  - 训练损失: {avg_train_loss:.4f}")
    logger.info(f"  - 评估损失: {eval_results['loss']:.4f}")
    logger.info(f"  - 学习率: {current_lr:.2e}")
    
    logger.info(f"\n整体评估指标:")
    logger.info(f"  - 准确率: {eval_results['precision']:.4f}")
    logger.info(f"  - 召回率: {eval_results['recall']:.4f}")
    logger.info(f"  - F1值: {eval_results['f1']:.4f}")

    # 输出每个标签的详细评估指标
    logger.info(f"\n各标签评估指标:")
    for label in label_list:
        if label == 'O':
            continue  # 跳过O标签
        if label in eval_results['report']:
            metrics = eval_results['report'][label]
            logger.info(f"\n{label}:")
            logger.info(f"  - 样本数: {metrics['support']}")
            logger.info(f"  - 准确率: {metrics['precision']:.4f}")
            logger.info(f"  - 召回率: {metrics['recall']:.4f}")
            logger.info(f"  - F1值: {metrics['f1-score']:.4f}")

    # 绘制训练过程中的指标变化图
    if len(train_history['train_losses']) > 0:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(train_history['train_losses'], label='训练损失')
        plt.plot(train_history['eval_losses'], label='验证损失')
        plt.legend()
        plt.title('损失曲线')

        plt.subplot(2, 2, 2)
        plt.plot(train_history['precisions'], label='准确率')
        plt.plot(train_history['recalls'], label='召回率')
        plt.plot(train_history['f1_scores'], label='F1值')
        plt.legend()
        plt.title('评估指标曲线')

        plt.subplot(2, 2, 3)
        plt.plot(train_history['learning_rates'], label='学习率')
        plt.legend()
        plt.title('学习率变化曲线')

        plt.tight_layout()
        plt.savefig(os.path.join(model_config['model_dir'], 'training_metrics.png'))
        plt.close()
    
    # 保存最佳模型
    if eval_results['f1'] > best_f1:
        best_f1 = eval_results['f1']
        torch.save(model.state_dict(), model_config['best_model_path'])
        
        # 保存最佳模型的分类报告
        report_path = os.path.join(model_config['model_dir'], 'best_classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Best Model Classification Report (Epoch {epoch + 1})\n")
            f.write(f"F1 Score: {best_f1:.4f}\n\n")
            f.write(json.dumps(eval_results['report'], indent=2, ensure_ascii=False))
        
        logger.info(f"\n✨ 保存新的最佳模型，F1: {best_f1:.4f}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        logger.info(f"\n⚠️ 未提升，EarlyStopping 计数: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            logger.info("\n🛑 提前停止训练（验证集 F1 无提升）")
            break

# 如果启用了模型剪枝，移除剪枝
if optimization_config['use_model_pruning']:
    logger.info("移除剪枝，使权重永久保持剪枝后的值...")
    model_optimizer.remove_pruning()

# 保存最终模型
torch.save(model.state_dict(), model_config['final_model_path'])
print(f"\n模型训练完成，最终模型已保存至 {model_config['final_model_path']}")

# 可视化训练损失
plt.figure()
loss_history_cpu = [loss.cpu().item() for loss in loss_history]
plt.plot(range(1, len(loss_history_cpu) + 1), loss_history_cpu, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.savefig(os.path.join(model_config['model_dir'], "training_loss.png"))

# 模型可解释性分析
if any([explainability_config['generate_attention_visualization'],
        explainability_config['generate_token_attention'],
        explainability_config['generate_prediction_confidence'],
        explainability_config['generate_html_visualization']]):
    
    logger.info("开始模型可解释性分析...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_config['best_model_path']))
    model.eval()
    
    # 初始化模型解释器
    explainer = ModelExplainer(model, tokenizer, id2label, device)
    
    # 从验证集中选择样本进行可视化
    with open(model_config['dev_path'], 'r', encoding='utf-8') as f:
        dev_samples = f.read().strip().split('\n\n')
    
    # 限制样本数量
    num_samples = min(explainability_config['visualization_samples'], len(dev_samples))
    selected_samples = random.sample(dev_samples, num_samples)
    
    for i, sample in enumerate(selected_samples):
        # 解析样本
        lines = sample.strip().split('\n')
        words = [line.split()[0] for line in lines if line.strip()]
        text = ''.join(words)
        
        # 创建样本目录
        sample_dir = os.path.join(explainability_config['visualization_output_dir'], f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存原始文本
        with open(os.path.join(sample_dir, "text.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        
        # 生成可视化
        if explainability_config['generate_attention_visualization']:
            explainer.visualize_attention(text, os.path.join(sample_dir, "attention_heatmap.png"))
        
        if explainability_config['generate_token_attention']:
            explainer.visualize_token_attention(text, os.path.join(sample_dir, "token_attention.png"))
        
        if explainability_config['generate_prediction_confidence']:
            explainer.visualize_prediction_confidence(text, os.path.join(sample_dir, "prediction_confidence.png"))
        
        if explainability_config['generate_html_visualization']:
            explainer.generate_html_visualization(text, os.path.join(sample_dir, "visualization.html"))
    
    logger.info(f"模型可解释性分析完成，结果保存在 {explainability_config['visualization_output_dir']}")

# 模型量化（如果启用）
if optimization_config['use_model_quantization']:
    logger.info("开始模型量化...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_config['best_model_path']))
    model.eval()
    
    try:
        # 量化模型
        quantized_model = model_optimizer.quantize_model(optimization_config['quantization_type'])
        
        # 保存量化模型
        quantized_model_path = os.path.join(model_config['model_dir'], "quantized_model.pth")
        model_optimizer.save_quantized_model(quantized_model_path, quantized_model)
        
        # 比较模型大小
        original_size, quantized_size = model_optimizer.compare_model_sizes(
            model_config['best_model_path'],
            quantized_model_path
        )
        
        # 保存量化结果
        quantization_result = {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "reduction_percentage": (1 - quantized_size / original_size) * 100
        }
        
        with open(os.path.join(model_config['model_dir'], "quantization_result.json"), "w") as f:
            json.dump(quantization_result, f, indent=2)
        
        logger.info(f"模型量化完成，量化模型已保存至 {quantized_model_path}")
    except Exception as e:
        logger.error(f"模型量化失败: {e}")

print("\n🎉 训练和优化流程全部完成!")