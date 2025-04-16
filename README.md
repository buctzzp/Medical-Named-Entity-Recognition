标签名	含义
O	Outside，非实体（其他普通词）
B-Anatomical	实体“解剖结构”的开头（如“肺”、“胃”、“大脑”）
I-Anatomical	解剖结构实体的中间部分
B-Diseases	实体“疾病”的开头（如“糖尿病”、“肺炎”）
I-Diseases	疾病实体的中间部分
B-Drug	实体“药品”的开头（如“阿莫西林”、“布洛芬”）
I-Drug	药品实体的中间部分
B-Image	实体“影像检查术语”的开头（如“CT”、“MRI”、“超声”）
I-Image	影像实体的中间部分
B-Laboratory	实体“实验室指标/检查”的开头（如“血糖”、“血压”、“肌酐”）
I-Laboratory	实验室实体的中间部分
B-Operation	实体“操作/手术名词”的开头（如“剖宫产”、“关节置换”）
I-Operation	操作实体的中间部分

本模型使用了 BIO 标注体系，共识别 6 类医学实体，包括 Anatomical（解剖结构）、Diseases（疾病）、Drug（药品）、Image（影像检查）、Laboratory（实验室指标）和 Operation（手术操作）。每类标签分为 B（开始）和 I（内部），共计 13 个标签。
