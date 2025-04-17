#!/usr/bin/env python
# evaluate.py — 模型评估脚本

import os
import torch
import argparse
import json
import logging
import sys  # 添加sys模块
from tqdm import tqdm
from datetime import datetime
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 尝试找到可用的中文字体
try:
    from matplotlib.font_manager import FontProperties
    # Windows常见中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 
                    'STKaiti', 'STSong', 'STFangsong', 'STXihei', 'STZhongsong']
    font_found = False
    for font in chinese_fonts:
        try:
            FontProperties(fname=font)
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            font_found = True
            print(f"使用中文字体: {font}")
            break
        except:
            continue
    
    if not font_found:
        print("警告: 未找到可用的中文字体，图表中的中文可能显示不正确")
except:
    print("警告: 配置中文字体失败，图表中的中文可能显示不正确")

from model.bert_crf_model import BertCRF
from model.bert_attention_crf_model import BertAttentionCRF
from model.model_factory import ModelFactory
from utils import NERDataset, get_label_list
from config import model_config

# 检查防止模块被多次导入执行
# 如果脚本直接运行而不是被导入，__name__会是'__main__'
if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='中文医疗NER模型评估工具')
    parser.add_argument('--model', type=str, default=model_config.get('best_model_path', 'models/best_model.pth'), 
                        help='模型权重路径')
    parser.add_argument('--pretrained_model', type=str, default=model_config.get('bert_model_name', 'bert-base-chinese'), 
                       help='预训练模型名称，需与训练时一致')
    parser.add_argument('--test_file', type=str, default=model_config.get('test_path', 'data/test.txt'), help='测试数据文件路径')
    parser.add_argument('--batch_size', type=int, default=model_config.get('eval_batch_size', 32), help='评估批次大小')
    parser.add_argument('--max_length', type=int, default=model_config.get('max_len', 128), help='最大序列长度')
    parser.add_argument('--use_attention', action='store_true', default=model_config.get('use_attention', False), 
                       help='是否使用注意力模型')
    parser.add_argument('--use_bilstm', action='store_true', default=model_config.get('use_bilstm', False), 
                       help='是否使用BiLSTM层')
    parser.add_argument('--no_bilstm', action='store_true', help='禁用BiLSTM层（覆盖默认配置）')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--detailed_report', action='store_true', help='输出详细的评估报告')
    parser.add_argument('--confusion_matrix', action='store_true', help='生成混淆矩阵')
    parser.add_argument('--save_predictions', action='store_true', help='保存预测结果')
    args = parser.parse_args()

    # no_bilstm 比 use_bilstm 优先级高
    use_bilstm = args.use_bilstm
    if args.no_bilstm:
        use_bilstm = False

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 生成模型ID和签名，与train_enhanced.py和predict_enhanced.py保持一致
    model_id = args.pretrained_model.replace('-', '_').replace('/', '_')
    model_type = "attention" if args.use_attention else "base"
    bilstm_status = "with_bilstm" if use_bilstm else "no_bilstm"
    model_signature = f"{model_id}_{model_type}_{bilstm_status}"

    # 根据参数选择模型路径
    # 使用安全路径，避免斜杠问题
    safe_model_id = model_id.replace('/', '_')
    model_dir = os.path.join(model_config.get('model_dir', 'models'), safe_model_id)
    os.makedirs(model_dir, exist_ok=True)

    if args.model == 'best':
        model_path = os.path.join(model_dir, f"best_model_{model_signature}.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, f"best_model_{model_id}.pth")  # 兼容旧版路径
            if not os.path.exists(model_path):
                model_path = model_config.get('best_model_path', 'models/best_model.pth')  # 回退到默认路径
    elif args.model == 'final':
        model_path = os.path.join(model_dir, f"final_model_{model_signature}.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, f"final_model_{model_id}.pth")  # 兼容旧版路径
            if not os.path.exists(model_path):
                model_path = model_config.get('final_model_path', 'models/final_model.pth')  # 回退到默认路径
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
                        model_path = model_config.get('best_model_path', 'models/best_model.pth')  # 再次回退
    else:
        # 用户指定的具体路径
        model_path = args.model

    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用安全路径，避免斜杠问题
    safe_model_id = model_id.replace('/', '_')
    eval_dir = os.path.join(args.output_dir, safe_model_id, f"eval_{model_signature}_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)

    # 设置日志
    log_file = os.path.join(eval_dir, f'evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 指定编码为utf-8
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 记录评估配置 (移除emoji)
    logger.info("="*50)
    logger.info("医学命名实体识别模型评估开始")
    logger.info("="*50)
    logger.info(f"预训练模型: {args.pretrained_model}")
    logger.info(f"模型类型: {'BERT-Attention-CRF' if args.use_attention else 'BERT-CRF'}")
    logger.info(f"BiLSTM层: {'启用' if use_bilstm else '禁用'}")
    logger.info(f"模型签名: {model_signature}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"测试文件: {args.test_file}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"最大长度: {args.max_length}")
    logger.info("="*50)

    # 控制台输出可以保留emoji
    print(f"🚀 使用预训练模型: {args.pretrained_model}")
    print(f"🔍 模型类型: {'BERT-Attention-CRF' if args.use_attention else 'BERT-CRF'}")
    print(f"🧠 BiLSTM层: {'启用' if use_bilstm else '禁用'}")
    print(f"📊 模型签名: {model_signature}")
    print(f"📁 评估结果将保存至: {eval_dir}")

    # 加载标签列表
    train_paths = []
    train_path = model_config.get('train_path', 'data/train.txt')
    if os.path.exists(train_path):
        train_paths.append(train_path)

    dev_path = model_config.get('dev_path', 'data/dev.txt')
    if os.path.exists(dev_path):
        train_paths.append(dev_path)

    # 如果没有找到训练文件，则使用测试文件来获取标签
    if not train_paths and os.path.exists(args.test_file):
        train_paths.append(args.test_file)

    if not train_paths:
        logger.error("未找到任何数据文件来获取标签列表，请检查数据路径")
        print("❌ 未找到任何数据文件来获取标签列表，请检查数据路径")
        sys.exit(1)  # 使用sys.exit替代exit

    label_list = get_label_list(train_paths)

    # 构建标签映射
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label_list)

    logger.info(f"标签总数: {num_labels}")
    logger.info(f"标签列表: {label_list}")

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
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"模型加载成功: {model_path}")
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        print(f"❌ 模型加载失败: {e}")
        print("请确保使用了与训练时相同的模型架构和预训练模型")
        sys.exit(1)  # 使用sys.exit替代exit

    model.to(device)
    model.eval()

    # 加载测试数据
    if not os.path.exists(args.test_file):
        logger.error(f"测试文件不存在: {args.test_file}")
        print(f"❌ 测试文件不存在: {args.test_file}")
        sys.exit(1)  # 使用sys.exit替代exit

    test_dataset = NERDataset(args.test_file, tokenizer, label2id, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             num_workers=4 if torch.cuda.is_available() else 0)

    logger.info(f"测试样本数: {len(test_dataset)}")
    print(f"📄 测试样本数: {len(test_dataset)}")

    # 辅助函数：确保所有值都是Python原生类型
    def convert_to_native_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native_python_types(obj.tolist())
        elif isinstance(obj, dict):
            return {key: convert_to_native_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_python_types(item) for item in obj]
        else:
            return obj

    # 评估函数
    def evaluate():
        model.eval()
        true_labels = []
        pred_labels = []
        all_sentences = []
        
        # 预测结果
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 原始文本（对于保存预测结果）
                tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch['input_ids']]
                
                # 模型预测
                pred = model(batch['input_ids'], batch['attention_mask'], 
                             batch.get('token_type_ids', None))
                
                # 处理批次中的每个样本
                for i in range(batch['input_ids'].size(0)):
                    label_ids = batch['labels'][i].tolist()
                    # 根据模型输出类型获取预测的标签ID
                    if isinstance(pred, list):
                        pred_ids = pred[i]
                    else:
                        pred_ids = pred[i].argmax(dim=-1).tolist()
                    
                    sample_tokens = tokens[i]
                    word_labels = []
                    word_preds = []
                    words = []
                    
                    for token, true_id, pred_id in zip(sample_tokens, label_ids, pred_ids):
                        if token in ["[CLS]", "[SEP]", "[PAD]"] or token.startswith("<") or true_id == -100:
                            continue
                        
                        # 保存实际的token
                        words.append(token.replace("##", ""))
                        word_labels.append(id2label[true_id])
                        word_preds.append(id2label[pred_id])
                    
                    if word_labels:
                        true_labels.append(word_labels)
                        pred_labels.append(word_preds)
                        all_sentences.append(words)
        
        # 计算整体指标
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        
        logger.info(f"整体性能: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        print(f"整体性能: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # 详细分类报告
        report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
        
        # 打印每种实体类型的详细指标
        entity_metrics = {}
        print("\n实体类型性能指标:")
        logger.info("\n实体类型性能指标:")
        for entity_type, metrics in report.items():
            if entity_type != "micro avg" and entity_type != "macro avg" and entity_type != "weighted avg" and isinstance(metrics, dict):
                entity_p = metrics['precision']
                entity_r = metrics['recall']
                entity_f1 = metrics['f1-score']
                support = metrics['support']
                entity_metrics[entity_type] = {
                    'precision': float(entity_p),
                    'recall': float(entity_r),
                    'f1': float(entity_f1),
                    'support': int(support)
                }
                print(f"  {entity_type}: P={entity_p:.4f}, R={entity_r:.4f}, F1={entity_f1:.4f}, 样本数={support}")
                logger.info(f"  {entity_type}: P={entity_p:.4f}, R={entity_r:.4f}, F1={entity_f1:.4f}, 样本数={support}")
        
        # 计算每个非O标签的实体类型
        entity_types = []
        for label in label_list:
            if label.startswith("B-") or label.startswith("I-"):
                entity_type = label[2:]  # 去掉"B-"或"I-"前缀
                if entity_type not in entity_types:
                    entity_types.append(entity_type)
        
        # 保存评估结果
        results = {
            'model_info': {
                'pretrained_model': args.pretrained_model,
                'model_signature': model_signature,
                'model_path': model_path,
                'use_attention': args.use_attention,
                'use_bilstm': use_bilstm
            },
            'eval_info': {
                'test_file': args.test_file,
                'num_samples': len(test_dataset),
                'timestamp': timestamp
            },
            'metrics': {
                'overall': {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                },
                'entity_metrics': entity_metrics
            }
        }
        
        # 确保所有值都是Python原生类型
        results = convert_to_native_python_types(results)
        
        # 保存JSON结果
        try:
            results_file = os.path.join(eval_dir, 'evaluation_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logger.info(f"评估结果已保存至: {results_file}")
        except Exception as e:
            logger.error(f"保存JSON结果时出错: {e}")
            print(f"❌ 保存JSON结果时出错: {e}")
        
        # 如果要保存详细报告
        if args.detailed_report:
            report_dir = os.path.join(args.output_dir, safe_model_id, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            try:
                detailed_report_file = os.path.join(report_dir, 'detailed_classification_report.txt')
                with open(detailed_report_file, 'w', encoding='utf-8') as f:
                    f.write(classification_report(true_labels, pred_labels, digits=4))
                logger.info(f"详细分类报告已保存至: {detailed_report_file}")
            except Exception as e:
                logger.error(f"保存详细分类报告时出错: {e}")
                print(f"❌ 保存详细分类报告时出错: {e}")
        
        # 如果要保存预测结果
        if args.save_predictions:
            try:
                predictions = []
                for i, (words, true_ls, pred_ls) in enumerate(zip(all_sentences, true_labels, pred_labels)):
                    predictions.append({
                        'id': i,
                        'words': words,
                        'true_labels': true_ls,
                        'pred_labels': pred_ls
                    })
                
                # 确保预测结果也使用Python原生类型
                predictions = convert_to_native_python_types(predictions)
                
                pred_file = os.path.join(eval_dir, 'predictions.json')
                with open(pred_file, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, ensure_ascii=False, indent=4)
                logger.info(f"预测结果已保存至: {pred_file}")
            except Exception as e:
                logger.error(f"保存预测结果时出错: {e}")
                print(f"❌ 保存预测结果时出错: {e}")
        
        # 绘制性能条形图
        try:
            plt.figure(figsize=(12, 6))
            metrics_df = pd.DataFrame([
                {'Entity Type': 'Overall', 'Precision': float(precision), 'Recall': float(recall), 'F1': float(f1)}
            ])
            
            for entity_type, metrics in entity_metrics.items():
                metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                    'Entity Type': entity_type,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1']
                }])], ignore_index=True)
            
            metrics_melted = pd.melt(metrics_df, id_vars=['Entity Type'], 
                                    value_vars=['Precision', 'Recall', 'F1'],
                                    var_name='Metric', value_name='Score')
            
            # 使用seaborn的设置提高图表美观度
            sns.set_style("whitegrid")
            # 设置颜色
            colors = sns.color_palette("muted", 3)
            # 绘制条形图
            ax = sns.barplot(data=metrics_melted, x='Entity Type', y='Score', hue='Metric', palette=colors)
            
            # 使用英文标签避免中文字体问题
            plt.title('NER Performance Metrics', fontsize=14, fontweight='bold')
            plt.xlabel('Entity Types', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Metrics', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 添加中文标题的说明（可选）
            plt.figtext(0.5, 0.01, "命名实体识别评估结果", ha="center", fontsize=10, alpha=0.7)
            
            chart_path = os.path.join(eval_dir, 'entity_performance.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            logger.info(f"性能条形图已保存至: {chart_path}")
        except Exception as e:
            logger.error(f"绘制性能条形图时出错: {e}")
            print(f"❌ 绘制性能条形图时出错: {e}")
        
        # 如果需要混淆矩阵
        if args.confusion_matrix and entity_types:
            cm_dir = os.path.join(args.output_dir, safe_model_id, "confusion_matrices")
            os.makedirs(cm_dir, exist_ok=True)
            
            try:
                # 准备混淆矩阵数据
                all_true_flat = []
                all_pred_flat = []
                
                for true_ls, pred_ls in zip(true_labels, pred_labels):
                    # 提取实体类型（去掉BIO前缀）
                    true_entity_types = ['O' if label == 'O' else label[2:] for label in true_ls]
                    pred_entity_types = ['O' if label == 'O' else label[2:] for label in pred_ls]
                    
                    all_true_flat.extend(true_entity_types)
                    all_pred_flat.extend(pred_entity_types)
                
                # 获取所有可能的实体类型（包括O）
                unique_entity_types = ['O'] + entity_types
                
                # 构建混淆矩阵
                confusion = np.zeros((len(unique_entity_types), len(unique_entity_types)))
                entity_to_idx = {entity: idx for idx, entity in enumerate(unique_entity_types)}
                
                for true_type, pred_type in zip(all_true_flat, all_pred_flat):
                    confusion[entity_to_idx[true_type]][entity_to_idx[pred_type]] += 1
                
                # 绘制混淆矩阵
                plt.figure(figsize=(10, 8))
                sns.set_style("white")
                
                # 使用英文标签避免中文字体问题
                ax = sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues',
                           xticklabels=unique_entity_types, yticklabels=unique_entity_types)
                
                plt.xlabel('Predicted Types', fontsize=12)
                plt.ylabel('True Types', fontsize=12)
                plt.title('Entity Type Confusion Matrix', fontsize=14, fontweight='bold')
                
                # 添加中文标题的说明（可选）
                plt.figtext(0.5, 0.01, "实体类型混淆矩阵", ha="center", fontsize=10, alpha=0.7)
                
                plt.tight_layout()
                
                confusion_path = os.path.join(cm_dir, 'confusion_matrix.png')
                plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
                logger.info(f"混淆矩阵已保存至: {confusion_path}")
            except Exception as e:
                logger.error(f"生成混淆矩阵时出错: {e}")
                print(f"❌ 生成混淆矩阵时出错: {e}")
        
        print(f"\n✅ 评估完成! 结果已保存至 {eval_dir}")
        logger.info(f"评估完成! 结果已保存至 {eval_dir}")
        
        return precision, recall, f1, report

    # 只有当脚本直接运行时才执行evaluate函数
    # 并且只执行一次
    evaluate()