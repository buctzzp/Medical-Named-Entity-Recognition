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
    def get_model_info(model_name):
        """
        获取模型信息，支持使用简称或完整名称
        
        Args:
            model_name: 模型名称（简称或完整路径）
            
        Returns:
            tuple: (model_path, model_type_family, short_name)
        """
        available_models = model_config['available_pretrained_models']
        
        # 检查是否是简称
        if model_name in available_models:
            model_info = available_models[model_name]
            return model_info['name'], model_info['type'], model_name
        
        # 检查是否是完整路径名称
        for short_name, info in available_models.items():
            if info['name'] == model_name:
                return model_name, info['type'], short_name
        
        # 如果都不是，则假设是未配置的huggingface模型路径
        return model_name, 'bert', None  # 默认使用BERT类型
    
    @staticmethod
    def create_model(model_type='bert_crf', num_labels=13, pretrained_model_name=None, use_attention=False):
        """
        创建NER模型
        
        Args:
            model_type: 模型类型，可选 'bert_crf' 或 'bert_attention_crf'
            num_labels: 标签数量
            pretrained_model_name: 预训练模型的名称或路径
            use_attention: 是否使用注意力机制
            
        Returns:
            初始化好的模型实例
        """
        if pretrained_model_name is None:
            pretrained_model_name = 'bert-base-chinese'
        
        # 获取预训练模型的具体配置
        model_path, model_type_family, _ = ModelFactory.get_model_info(pretrained_model_name)
        
        print(f"创建模型: {model_path}-{model_type}")
        
        # 根据模型类型创建相应的模型
        if use_attention or model_type == 'bert_attention_crf':
            model = BertAttentionCRF(model_path, num_labels)
            print(f"已选择 {model_path}-Attention-CRF 模型架构")
        else:
            model = BertCRF(model_path, num_labels)
            print(f"已选择 {model_path}-CRF 模型架构")
        
        return model
    
    @staticmethod
    def get_tokenizer_for_model(model_name):
        """
        获取与模型匹配的分词器
        
        Args:
            model_name: 模型名称或路径
            
        Returns:
            适合该模型的分词器
        """
        from transformers import AutoTokenizer, BertTokenizerFast, RobertaTokenizerFast
        
        # 获取预训练模型的具体配置
        model_path, model_type, _ = ModelFactory.get_model_info(model_name)
        
        if model_type == 'roberta':
            return RobertaTokenizerFast.from_pretrained(model_path)
        else:
            return BertTokenizerFast.from_pretrained(model_path)
            
    @staticmethod
    def list_available_models():
        """
        列出所有可用的预训练模型
        
        Returns:
            可用模型的列表，包括简称和完整路径
        """
        available_models = model_config['available_pretrained_models']
        # 返回所有简称和完整名称
        all_models = list(available_models.keys())
        all_models.extend([info['name'] for info in available_models.values()])
        return all_models 