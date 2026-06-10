import os

train_label_dir = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\train'
test_label_dir = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\test'

def check_labels(label_dir):
    invalid_files = []
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(label_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line_num, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"无效行 {filepath}:{line_num+1} 格式错误")
                invalid_files.append(filepath)
                break
            try:
                class_id, xc, yc, w, h = map(float, parts)
            except:
                print(f"无效行 {filepath}:{line_num+1} 转换失败")
                invalid_files.append(filepath)
                break
            if w <= 0 or h <= 0 or xc < 0 or xc > 1 or yc < 0 or yc > 1:
                print(f"无效框 {filepath}:{line_num+1} 坐标超出范围")
                invalid_files.append(filepath)
                break
    return invalid_files

train_invalid = check_labels(train_label_dir)
test_invalid = check_labels(test_label_dir)

print(f"训练集无效文件数: {len(train_invalid)}")
print(f"测试集无效文件数: {len(test_invalid)}")

# 删除无效文件及其对应图片
for f in train_invalid:
    os.remove(f)
    img_file = f.replace('labels\\train', 'train\\img').replace('.txt', '.jpg')
    if os.path.exists(img_file):
        os.remove(img_file)
for f in test_invalid:
    os.remove(f)
    img_file = f.replace('labels\\test', 'test\\img').replace('.txt', '.jpg')
    if os.path.exists(img_file):
        os.remove(img_file)

print("清理完成，请重新生成文件列表。")