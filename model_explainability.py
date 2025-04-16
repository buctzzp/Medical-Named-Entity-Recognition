# model_explainability.py - 模型可解释性分析模块

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from transformers import BertTokenizerFast
from typing import List, Dict, Tuple, Optional
import os
from PIL import Image
from io import BytesIO
import base64

class ModelExplainer:
    """模型可解释性分析类，提供注意力可视化、标签概率分析等功能"""
    
    def __init__(self, model, tokenizer: BertTokenizerFast, id2label: Dict[int, str], device: torch.device):
        """初始化模型解释器
        
        Args:
            model: BERT-CRF模型
            tokenizer: BERT分词器
            id2label: 标签ID到标签名的映射
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = device
        self.model.eval()  # 设置为评估模式
    
    def get_attention_weights(self, text: str) -> Tuple[List[str], np.ndarray]:
        """获取模型对输入文本的注意力权重
        
        Args:
            text: 输入文本
            
        Returns:
            tokens列表和注意力权重矩阵
        """
        # 将文本切分为字符列表
        chars = list(text.strip())
        
        # 编码输入
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True
        )
        
        # 将编码移动到设备上
        encoding = {k: v.to(self.device) for k, v in encoding.items() if k != 'offset_mapping'}
        
        # 获取注意力权重
        with torch.no_grad():
            # 设置输出注意力权重
            self.model.bert.config.output_attentions = True
            outputs = self.model.bert(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                token_type_ids=encoding['token_type_ids']
            )
            attentions = outputs.attentions  # 形状为 (batch_size, num_heads, seq_len, seq_len)
        
        # 解码token
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        # 计算平均注意力权重（跨所有层和头）
        # attentions是一个元组，每个元素对应一层的注意力
        attention_weights = torch.mean(torch.stack([layer_attention[0] for layer_attention in attentions]), dim=(0, 1))
        
        return tokens, attention_weights.cpu().numpy()
    
    def visualize_attention(self, text: str, output_path: Optional[str] = None, layer_idx: int = -1, head_idx: Optional[int] = None):
        """可视化注意力权重
        
        Args:
            text: 输入文本
            output_path: 输出图像路径，如果为None则显示图像
            layer_idx: 要可视化的层索引，默认为最后一层
            head_idx: 要可视化的注意力头索引，如果为None则使用所有头的平均值
        """
        # 获取注意力权重
        tokens, attention_weights = self.get_attention_weights(text)
        
        # 过滤掉特殊token和padding
        valid_tokens = []
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in ["[PAD]", "[CLS]", "[SEP]"] and not token.startswith("##"):
                valid_tokens.append(token)
                valid_indices.append(i)
        
        # 提取有效token之间的注意力权重
        valid_attention = attention_weights[np.ix_(valid_indices, valid_indices)]
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            valid_attention,
            xticklabels=valid_tokens,
            yticklabels=valid_tokens,
            cmap="YlGnBu",
            annot=False,
            fmt=".2f"
        )
        plt.title(f"注意力权重可视化: {text}")
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"注意力可视化已保存至 {output_path}")
        else:
            plt.show()
    
    def visualize_token_attention(self, text: str, output_path: Optional[str] = None):
        """可视化每个token的注意力分布
        
        Args:
            text: 输入文本
            output_path: 输出图像路径，如果为None则显示图像
        """
        # 获取注意力权重
        tokens, attention_weights = self.get_attention_weights(text)
        
        # 过滤掉特殊token和padding
        valid_tokens = []
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in ["[PAD]", "[CLS]", "[SEP]"] and not token.startswith("##"):
                valid_tokens.append(token)
                valid_indices.append(i)
        
        # 提取有效token的注意力权重
        valid_attention = attention_weights[valid_indices, :]
        valid_attention = valid_attention[:, valid_indices]
        
        # 计算每个token的平均注意力
        token_attention = np.mean(valid_attention, axis=1)
        
        # 创建条形图
        plt.figure(figsize=(12, 6))
        plt.bar(valid_tokens, token_attention, color='skyblue')
        plt.xlabel('Token')
        plt.ylabel('平均注意力权重')
        plt.title(f"每个Token的平均注意力权重: {text}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Token注意力可视化已保存至 {output_path}")
        else:
            plt.show()
    
    def predict_with_confidence(self, text: str) -> Tuple[List[str], List[str], List[float]]:
        """预测带置信度的结果
        
        Args:
            text: 输入文本
            
        Returns:
            字符列表、预测标签列表和置信度列表的元组
        """
        # 将文本切分为字符列表
        chars = list(text.strip())
        
        # 编码输入
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True
        )
        
        # 将编码移动到设备上
        encoding = {k: v.to(self.device) for k, v in encoding.items() if k != 'offset_mapping'}
        
        # 获取模型输出
        with torch.no_grad():
            # 获取发射分数
            outputs = self.model.bert(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                token_type_ids=encoding['token_type_ids']
            )
            sequence_output = outputs.last_hidden_state
            
            # 如果使用BiLSTM
            if hasattr(self.model, 'use_bilstm') and self.model.use_bilstm:
                sequence_output, _ = self.model.lstm(sequence_output)
            
            # 应用dropout和分类器
            sequence_output = self.model.dropout(sequence_output)
            emissions = self.model.classifier(sequence_output)
            
            # 获取CRF预测
            mask = encoding['attention_mask'].bool()
            pred_tags = self.model.crf.decode(emissions, mask=mask)[0]
            
            # 计算每个位置的置信度（使用softmax获取概率分布）
            probs = torch.softmax(emissions, dim=-1)
            confidences = []
            for i, tag_id in enumerate(pred_tags):
                if i < len(chars) and encoding['attention_mask'][0, i] == 1:
                    confidences.append(probs[0, i, tag_id].item())
        
        # 解码标签
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        valid_chars = []
        valid_tags = []
        valid_confidences = []
        
        for token, tag_id, conf in zip(tokens, pred_tags, confidences):
            if token not in ["[PAD]", "[CLS]", "[SEP]"] and not token.startswith("##"):
                valid_chars.append(token)
                valid_tags.append(self.id2label[tag_id])
                valid_confidences.append(conf)
        
        return valid_chars[:len(chars)], valid_tags[:len(chars)], valid_confidences[:len(chars)]
    
    def visualize_prediction_confidence(self, text: str, output_path: Optional[str] = None):
        """可视化预测置信度
        
        Args:
            text: 输入文本
            output_path: 输出图像路径，如果为None则显示图像
        """
        # 获取预测结果和置信度
        chars, tags, confidences = self.predict_with_confidence(text)
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 设置颜色映射
        cmap = cm.get_cmap('YlGnBu')
        colors = [cmap(conf) for conf in confidences]
        
        # 创建条形图
        bars = plt.bar(range(len(chars)), confidences, color=colors)
        
        # 添加标签
        for i, (char, tag, conf) in enumerate(zip(chars, tags, confidences)):
            plt.text(i, conf + 0.02, f"{char}\n{tag}", ha='center', va='bottom', rotation=0, fontsize=10)
        
        # 设置坐标轴
        plt.xlabel('字符位置')
        plt.ylabel('预测置信度')
        plt.title(f"预测置信度可视化: {text}")
        plt.ylim(0, 1.1)  # 设置y轴范围
        plt.xticks(range(len(chars)), [""] * len(chars))  # 隐藏x轴刻度标签
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"预测置信度可视化已保存至 {output_path}")
        else:
            plt.show()
    
    def generate_html_visualization(self, text: str, output_path: str):
        """生成HTML格式的可视化结果
        
        Args:
            text: 输入文本
            output_path: 输出HTML文件路径
        """
        # 获取预测结果和置信度
        chars, tags, confidences = self.predict_with_confidence(text)
        
        # 生成HTML内容
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>NER预测可视化</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 800px; margin: 0 auto; }
                .token { display: inline-block; margin: 5px; padding: 5px; border-radius: 3px; position: relative; }
                .token-text { font-size: 18px; }
                .token-tag { font-size: 12px; margin-top: 5px; text-align: center; }
                .token-confidence { font-size: 10px; margin-top: 2px; text-align: center; color: #666; }
                .legend { margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px; }
                .legend-item { display: inline-block; margin-right: 15px; }
                .legend-color { display: inline-block; width: 15px; height: 15px; margin-right: 5px; vertical-align: middle; border-radius: 2px; }
                h1 { color: #333; }
                .input-text { margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>命名实体识别预测可视化</h1>
                <div class="input-text">
                    <strong>输入文本:</strong> {text}
                </div>
                <div class="visualization">
        """.format(text=text)
        
        # 添加实体标记
        entity_colors = {
            "O": "#f0f0f0",  # 非实体为浅灰色
        }
        
        # 收集所有实体类型并分配颜色
        for tag in tags:
            if tag != "O" and tag.startswith("B-"):
                entity_type = tag[2:]  # 提取实体类型
                if entity_type not in entity_colors:
                    # 为每种实体类型分配一个颜色
                    hue = hash(entity_type) % 360  # 使用哈希值确保相同实体类型有相同颜色
                    entity_colors[entity_type] = f"hsl({hue}, 70%, 80%)"  # 使用HSL颜色空间
        
        # 添加每个字符的可视化
        for char, tag, conf in zip(chars, tags, confidences):
            # 确定背景颜色
            if tag == "O":
                bg_color = entity_colors["O"]
            else:
                entity_type = tag[2:]  # 提取实体类型
                bg_color = entity_colors[entity_type]
            
            # 根据置信度调整透明度
            opacity = 0.3 + 0.7 * conf  # 置信度越高，透明度越低
            
            html_content += f"""
                <div class="token" style="background-color: {bg_color}; opacity: {opacity};">
                    <div class="token-text">{char}</div>
                    <div class="token-tag">{tag}</div>
                    <div class="token-confidence">{conf:.2f}</div>
                </div>
            """
        
        # 添加图例
        html_content += """
                </div>
                <div class="legend">
                    <strong>图例:</strong><br>
        """
        
        for entity_type, color in entity_colors.items():
            if entity_type == "O":
                label = "非实体"
            else:
                label = entity_type
            html_content += f"""
                    <div class="legend-item">
                        <span class="legend-color" style="background-color: {color};"></span>
                        <span>{label}</span>
                    </div>
            """
        
        # 关闭HTML
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # 写入HTML文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"HTML可视化已保存至 {output_path}")

# 测试代码
if __name__ == "__main__":
    # 简单测试
    from transformers import BertModel, BertTokenizerFast
    from torchcrf import CRF
    import torch.nn as nn
    
    # 创建一个简单的BERT-CRF模型
    class SimpleBertCRF(nn.Module):
        def __init__(self):
            super(SimpleBertCRF, self).__init__()
            self.bert = BertModel.from_pretrained("bert-base-chinese")
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, 10)  # 假设有10个标签
            self.crf = CRF(10, batch_first=True)
            self.use_bilstm = False
        
        def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            sequence_output = outputs.last_hidden_state
            sequence_output = self.dropout(sequence_output)
            emissions = self.classifier(sequence_output)
            
            if labels is not None:
                mask = attention_mask.bool()
                loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
                return loss
            else:
                mask = attention_mask.bool()
                pred_tags = self.crf.decode(emissions, mask=mask)
                return pred_tags
    
    # 初始化模型和分词器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model = SimpleBertCRF().to(device)
    
    # 创建标签映射
    id2label = {0: "O", 1: "B-Disease", 2: "I-Disease", 3: "B-Symptom", 4: "I-Symptom", 
                5: "B-Drug", 6: "I-Drug", 7: "B-Treatment", 8: "I-Treatment", 9: "B-Test"}
    
    # 初始化解释器
    explainer = ModelExplainer(model, tokenizer, id2label, device)
    
    # 测试文本
    test_text = "患者出现了严重的头痛和发热症状"
    
    # 测试注意力可视化
    try:
        os.makedirs("results", exist_ok=True)
        explainer.visualize_attention(test_text, "results/attention_heatmap.png")
        explainer.visualize_token_attention(test_text, "results/token_attention.png")
        explainer.visualize_prediction_confidence(test_text, "results/prediction_confidence.png")
        explainer.generate_html_visualization(test_text, "results/visualization.html")
        print("测试完成")
    except Exception as e:
        print(f"测试失败: {e}")