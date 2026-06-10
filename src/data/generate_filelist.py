import os

base_dir = r'.\data\SportsMOT\deepsportradar-DatasetNinja'
train_img_dir = os.path.join(base_dir, 'train', 'img')
train_label_dir = os.path.join(base_dir, 'labels', 'train')
test_img_dir = os.path.join(base_dir, 'test', 'img')
test_label_dir = os.path.join(base_dir, 'labels', 'test')

def generate_list(img_dir, label_dir, out_file):
    with open(out_file, 'w') as f:
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
            if os.path.exists(label_path):
                f.write(f"{img_path} {label_path}\n")

# 输出文件仍然放在 labels 文件夹下（覆盖旧文件）
out_train = os.path.join(base_dir, 'labels', 'train.txt')
out_test = os.path.join(base_dir, 'labels', 'test.txt')
generate_list(train_img_dir, train_label_dir, out_train)
generate_list(test_img_dir, test_label_dir, out_test)

print(f"生成完成！训练集列表: {out_train}，测试集列表: {out_test}")