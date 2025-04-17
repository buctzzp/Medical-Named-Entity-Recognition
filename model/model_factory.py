import torch
from transformers import AutoModel, BertModel, RobertaModel
from model.bert_crf_model import BertCRF
from model.bert_attention_crf_model import BertAttentionCRF
from config import model_config

class ModelFactory:
    """
    模型工厂类，用于创建和加载不同类型的预训练模型和NER模型架构
    支持各种医疗领域预训练模型
    """
    
    @staticmethod
    def create_model(model_type='bert_crf', num_labels=13, pretrained_model_name=None, use_attention=False):
        """
        创建NER模型
        
        Args:
            model_type: 模型类型，可选 'bert_crf' 或 'bert_attention_crf'
            num_labels: 标签数量
            pretrained_model_name: 预训练模型的名称，如果为None则使用配置中的默认值
            use_attention: 是否使用注意力机制
            
        Returns:
            初始化好的模型实例
        """
        if pretrained_model_name is None:
            pretrained_model_name = 'bert-base-chinese'
        
        # 获取预训练模型的具体配置
        available_models = model_config['available_pretrained_models']
        if pretrained_model_name in available_models:
            model_info = available_models[pretrained_model_name]
            model_path = model_info['name']
            model_type_family = model_info['type']
        else:
            # 如果指定的模型不在配置列表中，则使用默认配置
            model_path = pretrained_model_name
            model_type_family = 'bert'  # 默认使用BERT类型
        
        print(f"创建模型: {model_type}, 使用预训练模型: {model_path}")
        
        # 根据模型类型创建相应的模型
        if use_attention or model_type == 'bert_attention_crf':
            model = BertAttentionCRF(model_path, num_labels)
            print("已选择 BERT-Attention-CRF 模型架构")
        else:
            model = BertCRF(model_path, num_labels)
            print("已选择 BERT-CRF 模型架构")
        
        return model
    
    @staticmethod
    def get_tokenizer_for_model(model_name):
        """
        获取与模型匹配的分词器
        
        Args:
            model_name: 模型名称
            
        Returns:
            适合该模型的分词器
        """
        from transformers import AutoTokenizer, BertTokenizerFast, RobertaTokenizerFast
        
        # 获取预训练模型的具体配置
        available_models = model_config['available_pretrained_models']
        if model_name in available_models:
            model_info = available_models[model_name]
            model_path = model_info['name']
            model_type = model_info['type']
            
            if model_type == 'roberta':
                return RobertaTokenizerFast.from_pretrained(model_path)
            else:
                return BertTokenizerFast.from_pretrained(model_path)
        else:
            # 如果不在配置列表中，使用AutoTokenizer
            return AutoTokenizer.from_pretrained(model_name)
            
    @staticmethod
    def list_available_models():
        """
        列出所有可用的预训练模型
        
        Returns:
            可用模型的列表
        """
        available_models = model_config['available_pretrained_models']
        return list(available_models.keys()) 