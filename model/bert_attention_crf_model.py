# bert_attention_crf_model.py - 增强版BERT-CRF模型，添加自注意力机制

import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from config import model_config

class SelfAttention(nn.Module):
    """自注意力层，用于增强特征提取能力"""
    
    def __init__(self, hidden_size, attention_size=None, num_attention_heads=1, dropout_prob=0.1):
        """初始化自注意力层
        
        Args:
            hidden_size: 输入特征维度
            attention_size: 注意力维度，默认等于hidden_size
            num_attention_heads: 注意力头数量
            dropout_prob: Dropout概率
        """
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_size = attention_size if attention_size else hidden_size
        self.all_head_size = self.num_attention_heads * self.attention_size
        
        # 查询、键、值的线性变换
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # 输出投影
        self.output = nn.Linear(self.all_head_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def transpose_for_scores(self, x):
        """将张量重塑为多头注意力格式"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_size)
    
    def forward(self, hidden_states, attention_mask=None):
        """前向传播
        
        Args:
            hidden_states: 输入特征，形状为 (batch_size, seq_len, hidden_size)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
            
        Returns:
            注意力加权后的特征
        """
        # 线性变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 重塑为多头格式
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_size ** 0.5)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 扩展注意力掩码以适应多头格式
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask
        
        # 应用softmax获取注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # 应用注意力权重
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # 重塑回原始维度
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 输出投影
        output = self.output(context_layer)
        output = self.dropout(output)
        output = self.layer_norm(output + hidden_states)  # 残差连接和层归一化
        
        return output

class BertAttentionCRF(nn.Module):
    def __init__(self, bert_model_name: str, num_labels: int):
        """初始化BERT-Attention-CRF模型
        Args:
            bert_model_name: BERT预训练模型名称
            num_labels: 标签数量
        """
        super(BertAttentionCRF, self).__init__()
        
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 添加自注意力层
        self.use_self_attention = model_config.get('use_self_attention', True)
        if self.use_self_attention:
            self.self_attention = SelfAttention(
                hidden_size=hidden_size,
                attention_size=model_config.get('attention_size', hidden_size),
                num_attention_heads=model_config.get('num_attention_heads', 8),
                dropout_prob=model_config.get('attention_dropout', 0.1)
            )
        
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
        self.dropout = nn.Dropout(p=model_config.get('hidden_dropout', 0.1))
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        
        # CRF层
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        
        # 标签平滑
        self.label_smoothing = model_config['label_smoothing']
        
        # 保存注意力权重（用于可视化）
        self.attention_weights = None

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """前向传播
        Args:
            input_ids: 输入token的ID序列 (batch_size, seq_len)
            attention_mask: 掩码张量 (batch_size, seq_len)
            token_type_ids: 句子类型ID (batch_size, seq_len)
            labels: 标签序列 (batch_size, seq_len)
        Returns:
            训练时返回损失，预测时返回预测标签序列
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True  # 输出注意力权重
        )
        sequence_output = outputs.last_hidden_state
        self.attention_weights = outputs.attentions  # 保存注意力权重
        
        # 自注意力层增强特征
        if self.use_self_attention:
            sequence_output = self.self_attention(sequence_output, attention_mask)
        
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
    
    def get_attention_weights(self):
        """获取最近一次前向传播的注意力权重"""
        return self.attention_weights