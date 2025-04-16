import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import numpy as np  # 添加numpy导入

def analyze_label_distribution(file_path):
    # 初始化标签计数器
    label_counter = Counter()
    
    # 读取文件并统计标签
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                _, label = line.split()
                if label != 'O':  # 不统计O标签
                    label_counter[label] += 1
    
    # 转换为DataFrame以便于展示
    df = pd.DataFrame(list(label_counter.items()), columns=['标签', '数量'])
    df = df.sort_values('数量', ascending=False)
    
    # 计算每个标签的占比
    total = df['数量'].sum()
    df['占比'] = df['数量'].apply(lambda x: f'{(x/total*100):.2f}%')
    
    # 设置pandas显示选项
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    
    # 打印统计结果
    print('\n' + '='*50)
    print('标签分布统计结果：')
    print('='*50)
    print(df.to_string(index=False, justify='center'))
    print('='*50 + '\n')
    
    # 设置图表样式
    plt.style.use('bmh')  # 使用matplotlib内置的bmh样式
    plt.figure(figsize=(12, 6))
    
    # 绘制条形图
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))  # 使用Set3颜色映射
    bars = plt.bar(df['标签'], df['数量'], color=colors)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}\n({df["占比"].iloc[i]})',
                ha='center', va='bottom')
    
    # 设置标题和标签
    plt.title('实体标签分布统计', fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('标签类型', fontsize=12)
    plt.ylabel('出现次数', fontsize=12)
    
    # 调整x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    train_file = 'data/train.txt'
    analyze_label_distribution(train_file)