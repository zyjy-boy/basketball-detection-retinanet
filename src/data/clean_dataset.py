import os
import glob

# 配置路径（根据你的实际情况修改）
base_dir = r'.\data\SportsMOT\deepsportradar-DatasetNinja'
train_img_dir = os.path.join(base_dir, 'train', 'img')
train_label_dir = os.path.join(base_dir, 'labels', 'train')
test_img_dir = os.path.join(base_dir, 'test', 'img')
test_label_dir = os.path.join(base_dir, 'labels', 'test')

def has_ball(label_path):
    """检查标签文件中是否有类别1（篮球）的标注"""
    if not os.path.exists(label_path):
        return False
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == 1:
                return True
    return False

def clean_split(img_dir, label_dir):
    """清理一个数据划分，删除无篮球的样本"""
    removed = 0
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, label_file)
        if not has_ball(label_path):
            # 删除标签文件
            os.remove(label_path)
            # 尝试删除对应的图片（支持多种扩展名）
            base = os.path.splitext(label_file)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(img_dir, base + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"删除无篮球样本: {img_path}")
                    removed += 1
                    break
    return removed

print("清理训练集...")
removed_train = clean_split(train_img_dir, train_label_dir)
print("清理验证集...")
removed_test = clean_split(test_img_dir, test_label_dir)
print(f"训练集删除了 {removed_train} 个无篮球样本，验证集删除了 {removed_test} 个无篮球样本。")

# 重新生成 train.txt 和 test.txt
def generate_list(img_dir, label_dir, out_file):
    with open(out_file, 'w') as f:
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            base = os.path.splitext(img_file)[0]
            label_path = os.path.join(label_dir, base + '.txt')
            if os.path.exists(label_path):
                img_path = os.path.join(img_dir, img_file)
                f.write(f"{img_path} {label_path}\n")

train_list = os.path.join(base_dir, 'labels', 'train.txt')
test_list = os.path.join(base_dir, 'labels', 'test.txt')
generate_list(train_img_dir, train_label_dir, train_list)
generate_list(test_img_dir, test_label_dir, test_list)
print(f"已重新生成训练列表: {train_list}")
print(f"已重新生成验证列表: {test_list}")