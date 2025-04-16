# model_optimization.py - 模型优化模块

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
from config import model_config

class ModelOptimizer:
    """模型优化类，提供混合精度训练、模型量化和模型剪枝等功能"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """初始化模型优化器
        
        Args:
            model: PyTorch模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.scaler = GradScaler()  # 用于混合精度训练的梯度缩放器
    
    def train_step_amp(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
        """使用混合精度训练的单步训练
        
        Args:
            batch: 包含输入数据的字典
            optimizer: 优化器
            
        Returns:
            损失值
        """
        # 将数据移动到设备上
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 使用混合精度前向传播
        with torch.amp.autocast('cuda'):
            loss = self.model(**batch)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), model_config['max_grad_norm'])
        
        # 更新参数
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def apply_pruning(self, amount: float = 0.3, method: str = 'l1_unstructured'):
        """应用模型剪枝
        
        Args:
            amount: 剪枝比例，范围[0, 1]
            method: 剪枝方法，支持'l1_unstructured'和'random_unstructured'
        """
        # 对模型中的线性层和卷积层应用剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                if method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=amount)
                elif method == 'random_unstructured':
                    prune.random_unstructured(module, name='weight', amount=amount)
                else:
                    raise ValueError(f"不支持的剪枝方法: {method}")
    
    def remove_pruning(self):
        """移除剪枝，使权重永久性地保持剪枝后的值"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
    
    def quantize_model(self, quantization_type: str = 'dynamic'):
        """量化模型
        
        Args:
            quantization_type: 量化类型，支持'dynamic'和'static'
        
        Returns:
            量化后的模型
        """
        if not torch.cuda.is_available():
            print("警告: 在CPU上进行量化可能不会带来性能提升")
        
        # 确保模型处于评估模式
        self.model.eval()
        
        if quantization_type == 'dynamic':
            # 动态量化 - 适用于LSTM和Transformer模型
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,  # 要量化的模型
                {nn.Linear, nn.LSTM},  # 要量化的层类型
                dtype=torch.qint8  # 量化数据类型
            )
            return quantized_model
        
        elif quantization_type == 'static':
            # 静态量化 - 需要校准数据，这里简化处理
            model_fp32 = self.model
            model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model_fp32_prepared = torch.quantization.prepare(model_fp32)
            
            # 这里应该使用校准数据集进行校准
            # 简化处理，直接量化
            quantized_model = torch.quantization.convert(model_fp32_prepared)
            return quantized_model
        
        else:
            raise ValueError(f"不支持的量化类型: {quantization_type}")
    
    def save_quantized_model(self, model_path: str, quantized_model: nn.Module):
        """保存量化模型
        
        Args:
            model_path: 模型保存路径
            quantized_model: 量化后的模型
        """
        torch.save(quantized_model.state_dict(), model_path)
        print(f"量化模型已保存至 {model_path}")
    
    def compare_model_sizes(self, original_model_path: str, quantized_model_path: str) -> Tuple[float, float]:
        """比较原始模型和量化模型的大小
        
        Args:
            original_model_path: 原始模型路径
            quantized_model_path: 量化模型路径
            
        Returns:
            原始模型大小(MB)和量化模型大小(MB)的元组
        """
        original_size = os.path.getsize(original_model_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
        
        print(f"原始模型大小: {original_size:.2f} MB")
        print(f"量化模型大小: {quantized_size:.2f} MB")
        print(f"大小减少: {(1 - quantized_size / original_size) * 100:.2f}%")
        
        return original_size, quantized_size
    
    def visualize_pruning_effect(self, model_path: str, output_dir: str = "checkpoints"):
        """可视化剪枝效果
        
        Args:
            model_path: 模型路径
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 统计每层的权重分布
        weight_distributions = {}
        for name, param in state_dict.items():
            if 'weight' in name and param.dim() > 1:  # 只分析权重矩阵
                weight_distributions[name] = param.flatten().cpu().numpy()
        
        # 绘制权重分布直方图
        plt.figure(figsize=(15, 10))
        for i, (name, weights) in enumerate(weight_distributions.items()):
            if i >= 9:  # 最多显示9个层
                break
            plt.subplot(3, 3, i+1)
            plt.hist(weights, bins=50)
            plt.title(name)
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "weight_distribution.png"))
        plt.close()
        
        print(f"权重分布图已保存至 {os.path.join(output_dir, 'weight_distribution.png')}")

# 测试代码
if __name__ == "__main__":
    # 简单测试
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 初始化优化器
    model_optimizer = ModelOptimizer(model, device)
    
    # 测试剪枝
    print("应用剪枝前的参数数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model_optimizer.apply_pruning(amount=0.5)
    print("应用剪枝后的非零参数数量:", sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad))
    
    # 测试量化
    try:
        quantized_model = model_optimizer.quantize_model()
        print("量化成功")
    except Exception as e:
        print(f"量化失败: {e}")