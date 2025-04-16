import os
from typing import List, Tuple, Dict
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import torch

# 定义命名实体识别数据集类
class NERDataset(Dataset):
    # 初始化数据集，加载tokenizer、标签与ID映射、最大长度，并读取数据
    # 该类继承自torch.utils.data.Dataset
    #file_path: 你给的数据文件路径，如 data/train.txt
    #tokenizer: BERT分词器，用于把每个字转成token
    #label2id: 标签映射字典（比如 {"B-Disease": 1, "I-Disease": 2, "O": 0}）
    #max_len: 最大序列长度（默认是128）

    def __init__(self, file_path: str, tokenizer: BertTokenizerFast, label2id: Dict[str, int], max_len: int = 128):
        # 初始化数据集，加载tokenizer、标签与ID映射、最大长度，并读取数据
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.texts, self.labels = self._read_data(file_path)

    # 读取数据文件，返回文本和标签的列表
    #texts = [['糖','尿','病'], ['是','一','种']]
    #labels = [['B-Disease','I-Disease','I-Disease'], ['O','O','O']]
    def _read_data(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        texts, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if not line:
                    # 如果当前行为空且words不为空，则保存当前的words和tags
                    if words:
                        texts.append(words)
                        labels.append(tags)
                        words, tags = [], []
                else:
                    splits = line.split()
                    if len(splits) != 2:
                        continue  # 跳过格式错误的行
                    word, tag = splits
                    words.append(word)
                    tags.append(tag)
            if words:
                texts.append(words)
                labels.append(tags)
        return texts, labels

    # 返回数据集中样本的数量
    def __len__(self):
        return len(self.texts)

    # 获取指定索引的样本
    def __getitem__(self, idx):
        words = self.texts[idx]  # 获取文本
        labels = self.labels[idx]  # 获取标签
        # 对文本进行编码
        encoding = self.tokenizer(words,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        word_ids = encoding.word_ids()  # 获取word_ids
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(self.label2id["O"])  # 对应空白位置的标签
            elif word_idx != previous_word_idx:
                # 如果当前词与上一个词不同，则使用当前词的标签
                label_ids.append(self.label2id.get(labels[word_idx], self.label2id['O']))
            else:
                # 如果当前词与上一个词相同，则保持标签不变
                label_ids.append(self.label2id.get(labels[word_idx], self.label2id['O']))
            previous_word_idx = word_idx  # 更新上一个词的索引

        # 将编码结果转化为tensor，并返回
        encoding = {key: torch.tensor(val) for key, val in encoding.items() if key in ['input_ids', 'attention_mask', 'token_type_ids']}
        encoding['labels'] = torch.tensor(label_ids)
        return encoding

# 获取标签列表
def get_label_list(file_paths: List[str]) -> List[str]:
    label_set = set()
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    splits = line.strip().split()
                    if len(splits) == 2:
                        _, tag = splits
                        label_set.add(tag)  # 添加标签到集合中
    label_list = sorted(label_set)  # 将标签排序
    return label_list

# ===== 用于测试的示例代码（可选运行）=====
if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    # 获取所有标签
    label_list = get_label_list([
        "data/train.txt", "data/dev.txt", "data/test.txt"
    ])
    label2id = {label: i for i, label in enumerate(label_list)}

    # 输出标签数量和映射字典
    print(f"标签总数: {len(label_list)}")
    print("标签与索引映射如下：")
    for i, label in enumerate(label_list):
        print(f"{i:>2} : {label}")

    # 加载训练集数据i
    #自动调用 __getitem__(0)
    dataset = NERDataset("data/train.txt", tokenizer, label2id)
    #print(len(dataset))  # 会自动调用 __len__

    # 输出第一条样本内容
    sample = dataset[0]
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    labels = sample["labels"].tolist()

    print("输入的tokens与标签如下：")
    for t, l in zip(tokens, labels):
        print(f"{t:10} => {l}")
