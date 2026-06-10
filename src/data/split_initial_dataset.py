import os
import random
import shutil

# 原始训练列表文件
train_list = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\train.txt'

# 输出初始数据集的列表文件
output_train_list = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\train_initial.txt'
output_val_list = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\val_initial.txt'

# 目标规模
target_train_size = 660
target_val_size = 44  # 约 660/0.8*0.2 ≈ 44

# 读取所有样本
with open(train_list, 'r') as f:
    all_samples = [line.strip() for line in f if line.strip()]

print(f"原始总样本数: {len(all_samples)}")

# 随机抽样（固定随机种子保证可复现）
random.seed(42)
selected_samples = random.sample(all_samples, target_train_size + target_val_size)

# 划分训练和验证
train_samples = selected_samples[:target_train_size]
val_samples = selected_samples[target_train_size:]

# 写入新列表文件
with open(output_train_list, 'w') as f:
    for sample in train_samples:
        f.write(sample + '\n')
with open(output_val_list, 'w') as f:
    for sample in val_samples:
        f.write(sample + '\n')

print(f"已生成初始训练集: {len(train_samples)} 张")
print(f"已生成初始验证集: {len(val_samples)} 张")