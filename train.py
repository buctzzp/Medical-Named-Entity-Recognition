import os
import torch
import logging
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_scheduler
from model.bert_crf_model import BertCRF
from utils import NERDataset, get_label_list
import matplotlib.pyplot as plt
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from config import model_config
from datetime import datetime

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

# 加载分词器和数据
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

# 加载数据集
train_dataset = NERDataset(model_config['train_path'], tokenizer, label2id, model_config['max_len'])
dev_dataset = NERDataset(model_config['dev_path'], tokenizer, label2id, model_config['max_len'])

train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=model_config['eval_batch_size'])

# 初始化模型与优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertCRF(model_config['bert_model_name'], num_labels).to(device)

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
    with torch.no_grad():
        for batch in dev_loader:
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

    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    return p, r, f1

# 训练循环
best_f1 = 0
loss_history = []
early_stop_counter = 0

for epoch in range(model_config['num_epochs']):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config['num_epochs']}")
    
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
                f'Epoch: {epoch+1}/{model_config["num_epochs"]}, '
                f'Step: {step+1}/{len(train_loader)}, '
                f'Loss: {loss.item():.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    # 验证集评估
    p, r, f1 = evaluate_on_dev()
    print(f"\tDev Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("\t✅ 新最佳模型，已保存至 checkpoints/best_model.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"\t⚠️ 未提升，EarlyStopping 计数: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("\n🛑 提前停止训练（验证集 F1 无提升）")
            break

# 保存最后模型
torch.save(model.state_dict(), "checkpoints/bert_crf.pth")
print("\n模型训练完成，最终模型已保存至 checkpoints/bert_crf.pth")

# 可视化训练损失
plt.figure()
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.savefig("checkpoints/training_loss.png")
plt.show()