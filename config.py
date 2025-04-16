# 模型配置
model_config = {
    # 基础配置
    'bert_model_name': 'bert-base-chinese',  # 中文预训练模型
    'max_len': 128,  # 保持最大序列长度
    'batch_size': 8,  # 减小batch_size以提高训练稳定性
    'num_epochs': 20,
    'learning_rate': 5e-5,  # 增加学习率
    'early_stopping_patience': 5,  # 增加早停耐心值
    
    # 模型结构配置
    'use_bilstm': True,  # 启用BiLSTM层以增强特征提取能力
    'lstm_hidden_size': 256,  # 保持LSTM隐藏层大小
    'lstm_layers': 4,  # LSTM层数
    'lstm_dropout': 0.1,  # LSTM dropout率
    
    # 注意力机制配置
    'use_self_attention': False,  # 不使用自注意力机制以减少计算量
    'attention_size': 768,  # 减小注意力维度
    'num_attention_heads': 8,  # 减少注意力头数量
    'attention_dropout': 0.1,  # 注意力dropout率
    'hidden_dropout': 0.1,  # 隐藏层dropout率
    
    # 训练策略配置
    'warmup_ratio': 0.1,  # 学习率预热比例
    'weight_decay': 0.01,  # AdamW权重衰减
    'max_grad_norm': 1.0,  # 梯度裁剪阈值
    'label_smoothing': 0.05,  # 减小标签平滑系数
    'use_amp': False,  # 是否使用混合精度训练
    
    # 路径配置
    'train_path': 'data/train.txt',
    'dev_path': 'data/dev.txt',
    'test_path': 'data/test.txt',
    'model_dir': 'checkpoints',
    'best_model_path': 'checkpoints/best_model.pth',
    'final_model_path': 'checkpoints/bert_crf.pth',
    'log_dir': 'logs',
    
    # 日志配置
    'logging_steps': 100,  # 每多少步记录一次日志
    'save_steps': 1000,  # 每多少步保存一次模型
    
    # 评估配置
    'eval_batch_size': 32,  # 评估时的batch size
    'do_train': True,  # 是否进行训练
    'do_eval': True,  # 是否进行评估
    'do_predict': True,  # 是否进行预测
}

# 数据增强配置
augmentation_config = {
    'use_data_augmentation': True,  # 是否使用数据增强
    'use_synonym_replace': True,  # 是否使用同义词替换
    'use_back_translate': True,  # 是否使用回译
    'use_entity_replace': True,  # 是否使用实体替换
    'synonym_replace_ratio': 0.1,  # 同义词替换比例
    'entity_replace_ratio': 0.1,  # 实体替换比例
    'synonym_dict_path': 'data/synonym_dict.json',  # 同义词词典路径
    'entity_dict_path': 'data/entity_dict.json',  # 实体词典路径
    'augmentation_size_multiplier': 1.5,  # 增强后数据集大小倍数
}

# 实验跟踪配置
experiment_config = {
    'use_wandb': False,  # 是否使用wandb进行实验跟踪
    'project_name': 'bert-crf-ner',
    'run_name': None,  # 运行名称，None则自动生成
    'tags': ['chinese-ner', 'bert-crf'],
}

# 模型优化配置
optimization_config = {
    'use_model_pruning': False,  # 是否使用模型剪枝
    'pruning_amount': 0.3,  # 剪枝比例
    'pruning_method': 'l1_unstructured',  # 剪枝方法
    'use_model_quantization': False,  # 是否使用模型量化
    'quantization_type': 'dynamic',  # 量化类型
    'apply_pruning_epoch': 10,  # 在第几个epoch应用剪枝
}

# 模型可解释性配置
explainability_config = {
    'generate_attention_visualization': False,  # 是否生成注意力可视化
    'generate_token_attention': False,  # 是否生成token注意力可视化
    'generate_prediction_confidence': False,  # 是否生成预测置信度可视化
    'generate_html_visualization': False,  # 是否生成HTML可视化
    'visualization_samples': 5,  # 可视化样本数量
    'visualization_output_dir': 'results/visualization',  # 可视化输出目录
}