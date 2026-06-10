import os

# 新测试集的图片和标签目录
img_dir = r'.\data\SportsMOT\Tracer-basketball.v3i.yolov8\test\images'
label_dir = r'.\data\SportsMOT\Tracer-basketball.v3i.yolov8\test\labels'
output_file = r'.\data\SportsMOT\Tracer-basketball.v3i.yolov8\test.txt'

with open(output_file, 'w') as f:
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            label_file = base + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                img_path = os.path.join(img_dir, img_file)
                f.write(f"{img_path} {label_path}\n")
                print(f"已添加: {img_file}")