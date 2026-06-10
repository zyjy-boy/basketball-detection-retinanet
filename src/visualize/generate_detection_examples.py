import os
import re
import torch
import cv2
import numpy as np
from collections import defaultdict
from torchvision.models.detection import retinanet_resnet50_fpn
from src.datasets.basketball_dataset import BasketballDataset
from torchvision import transforms as T

# ==================== 配置参数 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 训练好的模型权重路径
weights_path = 'weights/best_model.pth'

# 验证集列表文件（包含所有验证图片路径和标签路径）
train_list = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\train.txt'

# 输出目录
output_dir = 'results/detection_sequence'
os.makedirs(output_dir, exist_ok=True)

# 检测置信度阈值
conf_threshold = 0.1
# ==================================================

# 加载模型
def load_model(weights_path, num_classes=2):
    model = retinanet_resnet50_fpn(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.cls_logits.in_channels
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    model.head.classification_head.num_classes = num_classes
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

model = load_model(weights_path)

# 图像预处理（用于模型推理）
transform = T.Compose([
    T.ToTensor(),
    T.Resize((640, 640)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_boxes_from_label(label_path, img_shape):
    """从标签文件读取真实框（YOLO格式转绝对坐标）"""
    boxes = []
    h, w = img_shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, xc, yc, bw, bh = map(float, parts)
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                boxes.append([x1, y1, x2, y2])
    return boxes

def group_by_prefix(list_file):
    """将图片按文件名前缀分组（假设文件名形如 prefix_数字.ext）"""
    groups = defaultdict(list)
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, label_path = line.split()
            filename = os.path.basename(img_path)
            match = re.match(r'(.*)_(\d+)\.(png|jpg|jpeg)', filename, re.IGNORECASE)
            if match:
                prefix = match.group(1)
                frame_num = int(match.group(2))
                groups[prefix].append((frame_num, img_path, label_path))
            else:
                # 如果不匹配模式，则单独成组
                groups[filename].append((0, img_path, label_path))
    # 每组按帧号排序
    for prefix in groups:
        groups[prefix].sort(key=lambda x: x[0])
    return groups

print("正在解析验证集列表...")
groups = group_by_prefix(train_list)
print(f"共找到 {len(groups)} 个视频片段组")

# 处理每组的前几个连续帧
max_frames_per_group = 5
for prefix, frames in groups.items():
    if len(frames) < 2:
        continue  # 需要至少两帧才能画轨迹

    print(f"\n处理组: {prefix}, 共 {len(frames)} 帧")
    selected = frames[:max_frames_per_group]

    images_info = []  # 存储每帧的信息
    for frame_num, img_path, label_path in selected:
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图片 {img_path}，跳过")
            continue

        # 获取真实框
        gt_boxes = get_boxes_from_label(label_path, image.shape)

        # 模型预测
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)[0]

        # 筛选篮球（类别1），置信度高于阈值
        mask = (pred['labels'] == 1) & (pred['scores'] > conf_threshold)
        pred_boxes = pred['boxes'][mask].cpu().numpy()

        # 计算预测框中心点（用于轨迹连线）
        centers = []
        for box in pred_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            centers.append((cx, cy))

        images_info.append({
            'frame': frame_num,
            'img': image,
            'gt_boxes': gt_boxes,
            'pred_boxes': pred_boxes,
            'centers': centers
        })

    if len(images_info) < 2:
        continue  # 有效帧不足两帧，跳过

    # 绘制每一帧的检测结果并保存
    for info in images_info:
        img = info['img'].copy()
        # 画真实框（红色）
        for box in info['gt_boxes']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 画预测框（绿色）
        for box in info['pred_boxes']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 保存单帧图片
        out_path = os.path.join(output_dir, f"{prefix}_frame{info['frame']}.jpg")
        cv2.imwrite(out_path, img)
        print(f"已保存: {out_path}")

    # 生成水平拼接图并绘制轨迹
    # 将所有帧缩放到相同高度
    max_height = max(info['img'].shape[0] for info in images_info)
    resized_frames = []
    for info in images_info:
        h, w = info['img'].shape[:2]
        scale = max_height / h
        new_w = int(w * scale)
        resized = cv2.resize(info['img'], (new_w, max_height))
        # 在缩放后的图像上绘制框（直接在resized上绘制，避免再次缩放）
        # 需要重新计算框在resized中的坐标
        # 简便起见，我们使用原始框并绘制在resized上（但坐标需按比例缩放）
        # 这里我们简化：在原始图像上绘制好框再缩放（或直接在resized上绘制）
        # 为简化代码，我们在缩放后的图像上重新绘制框（通过遍历info中已经存储的框，但坐标需缩放）
        # 更好的做法：在原始图像上绘制好框，然后整体缩放，但缩放后框的坐标也会变化，可能模糊。
        # 我们采用另一种方法：将原始图像直接缩放，然后绘制框（框坐标按比例缩放）
        # 这里为了代码简洁，我们假设我们不需要在拼接图上画框，只画轨迹线。
        # 但为了显示效果，我们可以在缩放后的图像上重新绘制框（利用缩放后的坐标）
        # 我们来计算缩放后的框：
        gt_boxes_scaled = []
        for box in info['gt_boxes']:
            x1, y1, x2, y2 = box
            x1_scaled = x1 * scale
            y1_scaled = y1 * scale
            x2_scaled = x2 * scale
            y2_scaled = y2 * scale
            gt_boxes_scaled.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
        pred_boxes_scaled = []
        for box in info['pred_boxes']:
            x1, y1, x2, y2 = box
            x1_scaled = x1 * scale
            y1_scaled = y1 * scale
            x2_scaled = x2 * scale
            y2_scaled = y2 * scale
            pred_boxes_scaled.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])

        # 在resized图像上绘制框
        for box in gt_boxes_scaled:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for box in pred_boxes_scaled:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 保存缩放后的图像信息，同时记录中心点缩放后的坐标
        info['resized'] = resized
        info['scale'] = scale
        info['centers_scaled'] = [(cx * scale, cy * scale) for cx, cy in info['centers']]

    # 水平拼接
    combined = np.hstack([info['resized'] for info in images_info])

    # 绘制轨迹线：连接相邻帧的第一个预测框中心（假设只有一个篮球）
    centers_mapped = []
    offset_x = 0
    for i, info in enumerate(images_info):
        if info['centers_scaled']:
            # 取第一个预测框的中心
            cx, cy = info['centers_scaled'][0]
            mapped_cx = int(offset_x + cx)
            mapped_cy = int(cy)
            centers_mapped.append((mapped_cx, mapped_cy))
        offset_x += info['resized'].shape[1]

    # 绘制中心点（蓝色圆点）和连线
    for i, (cx, cy) in enumerate(centers_mapped):
        cv2.circle(combined, (cx, cy), 5, (255, 0, 0), -1)
        if i > 0:
            cv2.line(combined, centers_mapped[i-1], centers_mapped[i], (255, 0, 0), 2)

    # 保存组合轨迹图
    comb_path = os.path.join(output_dir, f"{prefix}_trajectory.jpg")
    cv2.imwrite(comb_path, combined)
    print(f"已保存轨迹图: {comb_path}")

print(f"\n所有处理完成！请查看 {output_dir} 文件夹中的图片。")