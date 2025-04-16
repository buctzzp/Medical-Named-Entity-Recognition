# 增强版NER评估脚本使用说明

## 功能概述

增强版的`evaluate.py`脚本提供了更灵活、全面的命名实体识别(NER)模型评估功能，主要改进包括：

1. **配置统一化**：使用`config.py`中的配置参数，便于统一管理和修改
2. **命令行参数支持**：可以通过命令行灵活指定评估数据集、模型路径等参数
3. **批量评估**：支持批量处理数据，提高评估效率
4. **增强评估报告**：提供更详细的评估指标和错误分析
5. **混淆矩阵可视化**：直观展示各类别之间的预测情况

## 使用方法

### 基本用法

```bash
python evaluate.py
```

这将使用默认参数（从`config.py`中读取）进行评估。

### 命令行参数

```bash
python evaluate.py --data data/test.txt --model checkpoints/best_model.pth --batch_size 16 --output results --confusion_matrix
```

参数说明：

- `--data`：评估数据集路径，默认为`config.py`中的`test_path`
- `--model`：模型路径，默认为`config.py`中的`final_model_path`
- `--batch_size`：评估批次大小，默认为`config.py`中的`eval_batch_size`
- `--output`：输出目录，默认为`results`
- `--confusion_matrix`：是否生成混淆矩阵，默认不生成

## 输出文件

评估脚本会在指定的输出目录生成以下文件：

1. `test_predictions.txt`：简洁版预测结果，每行包含字符和对应的预测标签
2. `test_predictions_verbose.txt`：详细版预测结果，包含位置、字符、真实标签、预测标签和是否正确的标记
3. `classification_report.txt`：分类报告，包含每个标签的精确率、召回率、F1分数等指标
4. `confusion_matrix.png`：混淆矩阵图（如果启用了`--confusion_matrix`选项）

## 评估指标

脚本会计算并输出以下评估指标：

- 每个实体类别的精确率、召回率、F1分数和支持度
- 总体的精确率、召回率和F1分数
- 错误分析统计，包括总标签数、错误预测数和错误率

## 示例输出

```
================== 分类指标报告 ==================
              precision    recall  f1-score   support

    B-Disease     0.8765    0.9012    0.8887       203
    I-Disease     0.9123    0.8954    0.9037       421
            O     0.9876    0.9934    0.9905      5421

    accuracy                         0.9823      6045
   macro avg     0.9255    0.9300    0.9276      6045
weighted avg     0.9789    0.9823    0.9805      6045

总体指标 - 精确率: 0.9255, 召回率: 0.9300, F1分数: 0.9276

错误分析统计:
总标签数: 6045
错误预测数: 107
错误率: 0.0177 (107/6045)
```

## 注意事项

1. 确保已安装所需的依赖包：`torch`, `transformers`, `seqeval`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`
2. 混淆矩阵生成可能需要较长时间，特别是对于标签数量较多的情况
3. 对于大型数据集，建议适当增加批次大小以提高评估速度