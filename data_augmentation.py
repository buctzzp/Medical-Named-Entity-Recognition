# data_augmentation.py - 数据增强模块

import random
import json
import os
import jieba
from typing import List, Dict, Tuple

class DataAugmentation:
    """数据增强类，提供多种NER数据增强方法"""
    
    def __init__(self, synonym_dict_path=None, entity_dict_path=None):
        """初始化数据增强类
        
        Args:
            synonym_dict_path: 同义词词典路径，格式为json，{"词":"同义词列表"}
            entity_dict_path: 实体词典路径，格式为json，{"实体类型":["实体列表"]}
        """
        # 加载同义词词典
        self.synonym_dict = {}
        if synonym_dict_path and os.path.exists(synonym_dict_path):
            with open(synonym_dict_path, 'r', encoding='utf-8') as f:
                self.synonym_dict = json.load(f)
        
        # 加载实体词典
        self.entity_dict = {}
        if entity_dict_path and os.path.exists(entity_dict_path):
            with open(entity_dict_path, 'r', encoding='utf-8') as f:
                self.entity_dict = json.load(f)
    
    def synonym_replace(self, words: List[str], labels: List[str], replace_ratio=0.1) -> Tuple[List[str], List[str]]:
        """同义词替换
        
        Args:
            words: 词列表
            labels: 标签列表
            replace_ratio: 替换比例
            
        Returns:
            替换后的词列表和标签列表
        """
        if not self.synonym_dict:
            return words, labels
        
        new_words = words.copy()
        new_labels = labels.copy()
        n_replace = max(1, int(len(words) * replace_ratio))
        
        # 找出所有可替换的位置（非实体部分）
        replaceable_indices = []
        i = 0
        while i < len(labels):
            if labels[i].startswith('O'):
                # 找到连续的O标签
                start = i
                while i < len(labels) and labels[i].startswith('O'):
                    i += 1
                end = i
                # 如果连续O标签组成的词长度大于1，则可以替换
                if end - start > 1:
                    text = ''.join(words[start:end])
                    # 使用jieba分词
                    segments = list(jieba.cut(text))
                    pos = start
                    for seg in segments:
                        if len(seg) > 1 and seg in self.synonym_dict and self.synonym_dict[seg]:
                            replaceable_indices.append((pos, pos + len(seg)))
                        pos += len(seg)
            else:
                i += 1
        
        # 随机选择n_replace个位置进行替换
        if replaceable_indices:
            random.shuffle(replaceable_indices)
            for start, end in replaceable_indices[:n_replace]:
                original_word = ''.join(new_words[start:end])
                if original_word in self.synonym_dict and self.synonym_dict[original_word]:
                    synonyms = self.synonym_dict[original_word]
                    synonym = random.choice(synonyms)
                    # 替换词
                    new_chars = list(synonym)
                    new_words[start:end] = new_chars
                    # 标签保持为O
                    new_labels[start:end] = ['O'] * len(new_chars)
        
        return new_words, new_labels
    
    def entity_replace(self, words: List[str], labels: List[str], replace_ratio=0.1) -> Tuple[List[str], List[str]]:
        """实体替换
        
        Args:
            words: 词列表
            labels: 标签列表
            replace_ratio: 替换比例
            
        Returns:
            替换后的词列表和标签列表
        """
        if not self.entity_dict:
            return words, labels
        
        new_words = words.copy()
        new_labels = labels.copy()
        
        # 找出所有实体
        entities = []
        i = 0
        while i < len(labels):
            if labels[i].startswith('B-'):
                entity_type = labels[i][2:]  # 获取实体类型
                start = i
                i += 1
                # 找到实体结束位置
                while i < len(labels) and labels[i].startswith('I-'):
                    i += 1
                end = i
                entities.append((start, end, entity_type))
            else:
                i += 1
        
        # 随机选择实体进行替换
        n_replace = max(1, int(len(entities) * replace_ratio))
        if entities and n_replace > 0:
            random.shuffle(entities)
            for start, end, entity_type in entities[:n_replace]:
                if entity_type in self.entity_dict and self.entity_dict[entity_type]:
                    # 随机选择同类型的实体进行替换
                    new_entity = random.choice(self.entity_dict[entity_type])
                    new_entity_chars = list(new_entity)
                    
                    # 替换实体
                    old_len = end - start
                    new_len = len(new_entity_chars)
                    
                    # 构建新的标签序列
                    new_entity_labels = ['I-' + entity_type] * new_len
                    new_entity_labels[0] = 'B-' + entity_type
                    
                    # 替换词和标签
                    new_words[start:end] = new_entity_chars
                    new_labels[start:end] = new_entity_labels[:old_len]
                    
                    # 如果新实体更长，需要插入额外的字符
                    if new_len > old_len:
                        for i in range(old_len, new_len):
                            new_words.insert(end, new_entity_chars[i])
                            new_labels.insert(end, new_entity_labels[i])
                            end += 1
                    # 如果新实体更短，不需要特殊处理，因为我们已经替换了原有位置
        
        return new_words, new_labels
    
    def augment_data(self, words: List[str], labels: List[str], 
                     use_synonym_replace=True, use_entity_replace=True,
                     synonym_replace_ratio=0.1, entity_replace_ratio=0.1) -> Tuple[List[str], List[str]]:
        """综合数据增强
        
        Args:
            words: 词列表
            labels: 标签列表
            use_synonym_replace: 是否使用同义词替换
            use_entity_replace: 是否使用实体替换
            synonym_replace_ratio: 同义词替换比例
            entity_replace_ratio: 实体替换比例
            
        Returns:
            增强后的词列表和标签列表
        """
        augmented_words, augmented_labels = words.copy(), labels.copy()
        
        # 同义词替换
        if use_synonym_replace and self.synonym_dict:
            augmented_words, augmented_labels = self.synonym_replace(
                augmented_words, augmented_labels, replace_ratio=synonym_replace_ratio)
        
        # 实体替换
        if use_entity_replace and self.entity_dict:
            augmented_words, augmented_labels = self.entity_replace(
                augmented_words, augmented_labels, replace_ratio=entity_replace_ratio)
        
        return augmented_words, augmented_labels

# 创建同义词和实体词典的辅助函数
def create_sample_dictionaries(output_dir="data"):
    """创建示例的同义词和实体词典"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例同义词词典
    synonym_dict = {
        "医院": ["医疗机构", "诊所", "医疗中心"],
        "疾病": ["病症", "疾患", "病情"],
        "治疗": ["医治", "诊治", "疗法"],
        "症状": ["病症", "症候", "表现"],
        "患者": ["病人", "病患", "就诊者"],
        "医生": ["医师", "大夫", "医务人员"],
        "药物": ["药品", "药剂", "药"],
    }
    
    # 示例实体词典
    entity_dict = {
        "Disease": ["糖尿病", "高血压", "肺炎", "感冒", "心脏病", "哮喘", "肝炎"],
        "Symptom": ["头痛", "发热", "咳嗽", "乏力", "腹痛", "恶心", "呕吐"],
        "Drug": ["阿司匹林", "青霉素", "布洛芬", "胰岛素", "维生素C", "泼尼松"],
        "Treatment": ["手术", "放疗", "化疗", "物理治疗", "针灸", "按摩"],
        "Test": ["血常规", "尿常规", "CT", "核磁共振", "X光", "超声波"],
        "Body": ["头部", "胸部", "腹部", "四肢", "心脏", "肺部", "肝脏"],
    }
    
    # 保存词典
    with open(os.path.join(output_dir, "synonym_dict.json"), "w", encoding="utf-8") as f:
        json.dump(synonym_dict, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "entity_dict.json"), "w", encoding="utf-8") as f:
        json.dump(entity_dict, f, ensure_ascii=False, indent=2)
    
    print(f"示例词典已保存至 {output_dir} 目录")

# 测试代码
if __name__ == "__main__":
    # 创建示例词典
    create_sample_dictionaries()
    
    # 测试数据增强
    augmenter = DataAugmentation(
        synonym_dict_path="data/synonym_dict.json",
        entity_dict_path="data/entity_dict.json"
    )
    
    # 测试样例
    words = list("患者出现了严重的头痛和发热症状，医生诊断为感冒")
    labels = ["O", "O", "O", "O", "O", "O", "O", "B-Symptom", "I-Symptom", "O", "B-Symptom", "I-Symptom", "O", "O", "O", "O", "O", "O", "O", "B-Disease", "I-Disease"]
    
    # 测试同义词替换
    new_words, new_labels = augmenter.synonym_replace(words, labels, replace_ratio=0.5)
    print("同义词替换结果:")
    print("".join(new_words))
    print(new_labels)
    
    # 测试实体替换
    new_words, new_labels = augmenter.entity_replace(words, labels, replace_ratio=1.0)
    print("\n实体替换结果:")
    print("".join(new_words))
    print(new_labels)
    
    # 测试综合增强
    new_words, new_labels = augmenter.augment_data(words, labels)
    print("\n综合增强结果:")
    print("".join(new_words))
    print(new_labels)