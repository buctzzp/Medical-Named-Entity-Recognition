# train_enhanced.py - 增强版训练脚本，整合所有优化功能

import os
import torch
import logging
import argparse
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
from config import model_config, augmentation_config, optimization_config, explainability_config, experiment_config
from datetime import datetime
import json
from model.model_factory import ModelFactory
import multiprocessing
from multiprocessing import freeze_support

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='中文医疗NER模型训练工具')
    parser.add_argument('--pretrained_model', type=str, default=model_config['bert_model_name'], 
                        help='预训练模型名称，可选：bert-base-chinese, chinese-medical-bert, pcl-medbert, cmeee-bert, mc-bert, chinese-roberta-med')
    parser.add_argument('--use_attention', action='store_true', default=model_config.get('use_attention', False), help='是否使用注意力模型')
    parser.add_argument('--use_bilstm', action='store_true', default=model_config.get('use_bilstm', False), help='是否使用BiLSTM层')
    parser.add_argument('--no_bilstm', action='store_true', help='禁用BiLSTM层（覆盖默认配置）')
    parser.add_argument('--batch_size', type=int, default=model_config['batch_size'], help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=model_config['num_epochs'], help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=model_config['learning_rate'], help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_augmentation', action='store_true', default=augmentation_config['use_data_augmentation'], help='是否使用数据增强')
    parser.add_argument('--no_augmentation', action='store_true', help='禁用数据增强（覆盖默认配置）')
    parser.add_argument('--early_stopping', type=int, default=model_config['early_stopping_patience'], help='早停耐心值')
    parser.add_argument('--save_every_epoch', action='store_true', help='是否每个epoch保存模型')
    parser.add_argument('--lstm_hidden_size', type=int, default=model_config['lstm_hidden_size'], help='LSTM隐藏层大小')
    parser.add_argument('--lstm_layers', type=int, default=model_config['lstm_layers'], help='LSTM层数')
    parser.add_argument('--use_model_pruning', action='store_true', default=optimization_config['use_model_pruning'], help='是否使用模型剪枝')
    parser.add_argument('--use_model_quantization', action='store_true', default=optimization_config['use_model_quantization'], help='是否使用模型量化')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 命令行参数覆盖配置
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    early_stopping_patience = args.early_stopping
    lstm_hidden_size = args.lstm_hidden_size
    lstm_layers = args.lstm_layers

    # 注意：no_bilstm 比 use_bilstm 优先级高
    use_bilstm = args.use_bilstm
    if args.no_bilstm:
        use_bilstm = False

    # 注意：no_augmentation 比 use_augmentation 优先级高
    use_augmentation = args.use_augmentation
    if args.no_augmentation:
        use_augmentation = False

    # 模型剪枝和量化
    use_model_pruning = args.use_model_pruning
    use_model_quantization = args.use_model_quantization

    # 获取模型ID用于文件命名
    model_id = args.pretrained_model.replace('-', '_').replace('/', '_')
    model_type = "attention" if args.use_attention else "base"
    bilstm_status = "with_bilstm" if use_bilstm else "no_bilstm"
    aug_status = "augmented" if use_augmentation else "no_aug"
    model_signature = f"{model_id}_{model_type}_{bilstm_status}_{aug_status}"

    # 检测GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 是否检测到 GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🖥️ 当前 GPU 名称:", torch.cuda.get_device_name(0))
        print("🔥 当前设备:", device)
    else:
        print("❌ 当前使用的是 CPU")

    # 创建必要的目录
    # 修复路径处理，避免/被当作目录分隔符
    safe_model_id = model_id.replace('/', '_')
    results_dir = os.path.join('results', safe_model_id)
    model_dir = os.path.join(model_config['model_dir'], safe_model_id)
    log_dir = os.path.join(model_config['log_dir'], safe_model_id)
    
    # 删除错误的目录（如果存在）
    wrong_results_dir = os.path.join('results', model_id.split('/')[0])
    wrong_model_dir = os.path.join(model_config['model_dir'], model_id.split('/')[0])
    wrong_log_dir = os.path.join(model_config['log_dir'], model_id.split('/')[0])
    
    # 如果存在错误目录但没有子内容，则删除
    for wrong_dir in [wrong_results_dir, wrong_model_dir, wrong_log_dir]:
        if '/' in model_id and os.path.exists(wrong_dir):
            try:
                # 检查是否为空目录
                if not os.listdir(wrong_dir):
                    os.rmdir(wrong_dir)
                    print(f"已删除错误创建的空目录: {wrong_dir}")
            except:
                pass
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(explainability_config['visualization_output_dir'], exist_ok=True)

    # 设置日志 - 使用utf-8编码
    log_filename = f'train_{model_signature}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_filename), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 记录训练配置信息 - 移除emoji以避免编码问题
    logger.info("="*50)
    logger.info("医学命名实体识别模型训练开始")
    logger.info("="*50)
    logger.info(f"预训练模型: {args.pretrained_model}")
    logger.info(f"模型架构: {'BERT-Attention-CRF' if args.use_attention else 'BERT-CRF'}")
    logger.info(f"BiLSTM层: {'启用' if use_bilstm else '禁用'}")
    if use_bilstm:
        logger.info(f"   - 隐藏层大小: {lstm_hidden_size}")
        logger.info(f"   - LSTM层数: {lstm_layers}")
        logger.info(f"   - 丢弃率: {model_config['lstm_dropout']}")
    logger.info(f"数据增强: {'启用' if use_augmentation else '禁用'}")
    logger.info(f"训练参数:")
    logger.info(f"   - 批次大小: {batch_size}")
    logger.info(f"   - 训练轮数: {num_epochs}")
    logger.info(f"   - 学习率: {learning_rate}")
    logger.info(f"   - 早停耐心值: {early_stopping_patience}")
    logger.info(f"   - 每轮保存: {'启用' if args.save_every_epoch else '禁用'}")
    logger.info(f"   - 模型剪枝: {'启用' if use_model_pruning else '禁用'}")
    logger.info(f"   - 模型量化: {'启用' if use_model_quantization else '禁用'}")
    logger.info(f"模型特征签名: {model_signature}")
    logger.info(f"模型结果存储位置: {results_dir}")
    logger.info(f"模型保存位置: {model_dir}")
    logger.info("="*50)

    # 打印选择的模型
    print(f"使用预训练模型: {args.pretrained_model}")
    print(f"模型类型: {'BERT-Attention-CRF' if args.use_attention else 'BERT-CRF'}")
    print(f"BiLSTM层: {'启用' if use_bilstm else '禁用'}")
    print(f"数据增强: {'启用' if use_augmentation else '禁用'}")
    print(f"结果保存目录: {results_dir}")
    print(f"模型保存目录: {model_dir}")

    # 获取可用预训练模型列表
    available_models = ModelFactory.list_available_models()
    if args.pretrained_model not in available_models:
        logger.warning(f"警告: 所选模型 '{args.pretrained_model}' 不在预配置列表中。可用模型: {', '.join(available_models)}")
        logger.warning(f"尝试直接从Hugging Face加载模型...")
        print(f"警告: 所选模型 '{args.pretrained_model}' 不在预配置列表中。可用模型: {', '.join(available_models)}")
        print(f"尝试直接从Hugging Face加载模型...")

    # 使用模型工厂获取与模型匹配的tokenizer
    tokenizer = ModelFactory.get_tokenizer_for_model(args.pretrained_model)

    # 构建标签映射
    label_list = get_label_list([
        model_config['train_path'],
        model_config['dev_path']
    ])
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label_list)

    logger.info(f"标签总数: {num_labels}")
    logger.info(f"标签列表: {label_list}")

    # 数据增强
    if use_augmentation:
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
        augmented_train_path = os.path.join(results_dir, f'train_augmented_{model_signature}.txt')
        with open(augmented_train_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_data))
        
        # 更新训练数据路径
        train_path = augmented_train_path
    else:
        train_path = model_config['train_path']

    # 加载数据集
    train_dataset = NERDataset(train_path, tokenizer, label2id, model_config['max_len'])
    dev_dataset = NERDataset(model_config['dev_path'], tokenizer, label2id, model_config['max_len'])

    # Windows上设置num_workers=0以避免多进程问题
    num_workers = 0 if os.name == 'nt' else (4 if torch.cuda.is_available() else 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=model_config['eval_batch_size'], num_workers=num_workers)

    # 使用模型工厂创建模型
    model = ModelFactory.create_model(
        model_type='bert_attention_crf' if args.use_attention else 'bert_crf',
        num_labels=num_labels,
        pretrained_model_name=args.pretrained_model,
        use_attention=args.use_attention
    )

    # 确保模型使用正确的BiLSTM设置
    if hasattr(model, 'use_bilstm'):
        model.use_bilstm = use_bilstm
    if hasattr(model, 'lstm_hidden_size'):
        model.lstm_hidden_size = lstm_hidden_size
    if hasattr(model, 'lstm_layers'):
        model.lstm_layers = lstm_layers

    model.to(device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params/1e6:.2f}M, 可训练 {trainable_params/1e6:.2f}M")
    print(f"模型参数: 总计 {total_params/1e6:.2f}M, 可训练 {trainable_params/1e6:.2f}M")

    # 初始化模型优化器
    model_optimizer = ModelOptimizer(model, device)

    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=model_config['weight_decay']
    )

    # 学习率调度器（带预热）
    num_training_steps = len(train_loader) * num_epochs
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
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
                
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

        # 计算整体指标
        p = precision_score(true_labels, pred_labels)
        r = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        
        # 详细分类报告
        report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
        
        return p, r, f1, report

    # 训练循环
    logger.info(f"开始训练: 使用 {args.pretrained_model} 模型, {'带' if args.use_attention else '不带'}注意力, {'使用' if use_bilstm else '不使用'}BiLSTM")
    best_f1 = 0
    epoch_metrics = []
    loss_history = []
    f1_history = []
    early_stop_counter = 0
    patience = early_stopping_patience

    print("\n训练开始...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config['max_grad_norm'])
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
            
            # 定期记录训练信息
            if (step + 1) % model_config['logging_steps'] == 0:
                logger.info(
                    f'Epoch: {epoch+1}/{num_epochs}, '
                    f'Step: {step+1}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f}, '
                    f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
                )

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # 验证集评估
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
        print("正在验证...")
        p, r, f1, report = evaluate_on_dev()
        f1_history.append(f1)
        
        # 记录本轮各实体类型的指标
        epoch_result = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'precision': p,
            'recall': r,
            'f1': f1,
            'entity_metrics': {}
        }
        
        # 记录详细的分类报告
        logger.info(f"Epoch {epoch+1} 验证集指标: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")
        print(f"\tDev Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        
        # 记录每种实体类型的详细指标
        for entity_type, metrics in report.items():
            if entity_type != "micro avg" and entity_type != "macro avg" and entity_type != "weighted avg" and isinstance(metrics, dict):
                entity_p = metrics['precision']
                entity_r = metrics['recall']
                entity_f1 = metrics['f1-score']
                logger.info(f"实体类型 {entity_type}: P={entity_p:.4f}, R={entity_r:.4f}, F1={entity_f1:.4f}")
                print(f"\t实体 {entity_type}: P={entity_p:.4f}, R={entity_r:.4f}, F1={entity_f1:.4f}")
                epoch_result['entity_metrics'][entity_type] = {
                    'precision': entity_p,
                    'recall': entity_r,
                    'f1': entity_f1
                }
        
        epoch_metrics.append(epoch_result)
        
        # 每轮保存模型（如果启用）
        if args.save_every_epoch:
            epoch_model_path = os.path.join(model_dir, f"epoch_{epoch+1}_{model_signature}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            logger.info(f"Epoch {epoch+1} 模型已保存至 {epoch_model_path}")
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            model_save_path = os.path.join(model_dir, f"best_model_{model_signature}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"\t新最佳模型，已保存至 {model_save_path}")
            logger.info(f"新最佳模型 (F1={f1:.4f})，已保存至 {model_save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"\t未提升，EarlyStopping 计数: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("\n提前停止训练（验证集 F1 无提升）")
                logger.info(f"提前停止训练: {early_stop_counter} 轮未见提升")
                break

    # 保存最后模型
    final_model_path = os.path.join(model_dir, f"final_model_{model_signature}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n模型训练完成，最终模型已保存至 {final_model_path}")
    logger.info(f"训练完成，最终模型已保存至 {final_model_path}")

    # 可视化训练损失和F1
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(f1_history) + 1), f1_history, marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1")
    plt.grid(True)

    plt.tight_layout()
    metrics_chart_path = os.path.join(results_dir, f"training_metrics_{model_signature}.png")
    plt.savefig(metrics_chart_path)
    print(f"训练指标图已保存至 {metrics_chart_path}")
    logger.info(f"训练指标图已保存至 {metrics_chart_path}")

    # 保存训练指标记录
    metrics_json_path = os.path.join(results_dir, f"training_metrics_{model_signature}.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_info': {
                'pretrained_model': args.pretrained_model,
                'use_attention': args.use_attention,
                'use_bilstm': use_bilstm,
                'use_augmentation': use_augmentation,
                'batch_size': batch_size,
                'epochs': num_epochs,
                'learning_rate': learning_rate
            },
            'best_f1': best_f1,
            'epochs': epoch_metrics
        }, f, ensure_ascii=False, indent=4)
    logger.info(f"训练指标记录已保存至 {metrics_json_path}")

    # 打印最终结果
    print("\n训练完成!")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"最佳模型路径: {model_save_path}")
    print(f"最终模型路径: {final_model_path}")
    print(f"详细训练记录: {metrics_json_path}")

    # 如果启用了实验跟踪
    if experiment_config['use_wandb']:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            logger.warning("未安装wandb，无法结束实验跟踪。")
        except Exception as e:
            logger.warning(f"关闭wandb时出错: {e}")

    print("\n评估模型: python evaluate.py --model " + model_save_path + " --pretrained_model " + args.pretrained_model + 
        (" --use_attention" if args.use_attention else "") + (" --use_bilstm" if use_bilstm else ""))

    print("\n预测样例: python predict_enhanced.py --model " + model_save_path + " --pretrained_model " + args.pretrained_model + 
        (" --use_attention" if args.use_attention else "") + (" --use_bilstm" if use_bilstm else "") + 
        " --input '患者出现高血压和2型糖尿病，建议服用降压药。'")

if __name__ == "__main__":
    # 在Windows上使用多进程时必需
    freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()