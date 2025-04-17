import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, RobertaModel
from torchcrf import CRF
from config import model_config

class BertCRF(nn.Module):
    def __init__(self, bert_model_name: str, num_labels: int):
        """初始化BERT-BiLSTM-CRF模型
        Args:
            bert_model_name: BERT预训练模型名称
            num_labels: 标签数量
        """
        super(BertCRF, self).__init__()
        
        # 加载预训练模型
        available_models = model_config.get('available_pretrained_models', {})
        if bert_model_name in available_models:
            model_info = available_models[bert_model_name]
            model_path = model_info['name']
            model_type = model_info['type']
            
            # 根据模型类型加载相应的预训练模型
            if model_type == 'bert':
                self.bert = BertModel.from_pretrained(model_path)
            elif model_type == 'roberta':
                self.bert = RobertaModel.from_pretrained(model_path)
            else:
                self.bert = AutoModel.from_pretrained(model_path)
        else:
            # 如果不在配置中，则直接使用AutoModel加载
            self.bert = AutoModel.from_pretrained(bert_model_name)
        
        hidden_size = self.bert.config.hidden_size
        
        # 添加BiLSTM层（可选）
        self.use_bilstm = model_config['use_bilstm']
        if self.use_bilstm:
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=model_config['lstm_hidden_size'],
                num_layers=model_config['lstm_layers'],
                bidirectional=True,
                dropout=model_config['lstm_dropout'] if model_config['lstm_layers'] > 1 else 0,
                batch_first=True
            )
            classifier_input_size = model_config['lstm_hidden_size'] * 2
        else:
            classifier_input_size = hidden_size
        
        # 分类层和Dropout
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        
        # CRF层
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        
        # 标签平滑
        self.label_smoothing = model_config['label_smoothing']

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """前向传播
        Args:
            input_ids: 输入token的ID序列 (batch_size, seq_len)
            attention_mask: 掩码张量 (batch_size, seq_len)
            token_type_ids: 句子类型ID (batch_size, seq_len)，对于某些模型如RoBERTa可能不需要
            labels: 标签序列 (batch_size, seq_len)
        Returns:
            训练时返回损失，预测时返回预测标签序列
        """
        # 检查模型类型并适当调整参数
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # 某些模型如RoBERTa不使用token_type_ids
        if token_type_ids is not None and hasattr(self.bert, 'embeddings') and hasattr(self.bert.embeddings, 'token_type_embeddings'):
            model_inputs['token_type_ids'] = token_type_ids
        
        # 获取模型编码
        outputs = self.bert(**model_inputs)
        sequence_output = outputs.last_hidden_state
        
        # BiLSTM特征提取
        if self.use_bilstm:
            sequence_output, _ = self.lstm(sequence_output)
        
        # Dropout和分类
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # 训练阶段
            mask = attention_mask.bool()
            
            # 标签平滑处理
            if self.label_smoothing > 0:
                # 将one-hot标签转换为软标签
                num_labels = emissions.size(-1)
                smooth_labels = torch.zeros_like(emissions).scatter_(-1, labels.unsqueeze(-1), 1.0)
                smooth_labels = smooth_labels * (1 - self.label_smoothing) + self.label_smoothing / num_labels
                
                # 计算损失
                log_probs = torch.log_softmax(emissions, dim=-1)
                loss = -(smooth_labels * log_probs).sum(dim=-1)
                loss = (loss * mask).sum() / mask.sum()
            else:
                # 使用CRF损失
                loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            # 预测阶段
            mask = attention_mask.bool()
            pred_tags = self.crf.decode(emissions, mask=mask)
            return pred_tags
