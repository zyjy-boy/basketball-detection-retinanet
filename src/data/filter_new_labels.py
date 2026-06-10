import os

# 新数据集路径（请根据实际情况修改）
base_new = r'.\data\SportsMOT\Basketball Detection.v1i.yolov8'
train_label_dir = os.path.join(base_new, 'train', 'labels')
train_img_dir = os.path.join(base_new, 'train', 'images')
valid_label_dir = os.path.join(base_new, 'valid', 'labels')
valid_img_dir = os.path.join(base_new, 'valid', 'images')

def filter_labels(label_dir, img_dir):
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, filename)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id == 0:   # 只保留篮球
                # 将类别ID改为1（你的模型篮球类别为1）
                new_line = '1 ' + ' '.join(parts[1:]) + '\n'
                new_lines.append(new_line)
        if new_lines:
            # 写回原标签文件
            with open(label_path, 'w') as f:
                f.writelines(new_lines)
        else:
            # 无篮球标注，删除标签文件和对应图片
            os.remove(label_path)
            base = os.path.splitext(filename)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(img_dir, base + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"删除无篮球图片: {img_path}")
                    break

print("处理训练集...")
filter_labels(train_label_dir, train_img_dir)
print("处理验证集...")
filter_labels(valid_label_dir, valid_img_dir)
print("过滤完成！")