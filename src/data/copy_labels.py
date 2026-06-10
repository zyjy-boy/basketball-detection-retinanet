import os
import shutil

# ==================== 配置 ====================
# 场景图片的根目录（包含 near, far, bright, dark 四个子文件夹）
scenes_root = r'.\data\selected_images'

# 存放所有标签的文件夹（之前复制过来的）
labels_source = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\test'

# 场景列表（与子文件夹名称一致）
scenes = ['near', 'far', 'brigh', 'dark']
# =============================================

# 支持的图片扩展名
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

for scene in scenes:
    scene_img_dir = os.path.join(scenes_root, scene)
    if not os.path.exists(scene_img_dir):
        print(f"警告：场景文件夹 {scene_img_dir} 不存在，跳过")
        continue

    # 创建该场景的 labels 子目录
    scene_label_dir = os.path.join(scene_img_dir, 'labels')
    os.makedirs(scene_label_dir, exist_ok=True)

    # 遍历该场景下的所有图片（支持递归，但通常图片直接在场景文件夹下）
    for root, dirs, files in os.walk(scene_img_dir):
        # 跳过已经创建的 labels 目录，避免重复处理
        if os.path.basename(root) == 'labels':
            continue
        for file in files:
            if file.lower().endswith(img_exts):
                base = os.path.splitext(file)[0]
                label_file = base + '.txt'
                src_label = os.path.join(labels_source, label_file)
                if os.path.exists(src_label):
                    dst_label = os.path.join(scene_label_dir, label_file)
                    shutil.copy2(src_label, dst_label)
                    print(f"✅ 已复制 {label_file} 到 {scene}/labels/")
                else:
                    print(f"⚠️ 未找到标签: {label_file} (图片: {os.path.join(root, file)})")

print("\n所有标签分类完成！")