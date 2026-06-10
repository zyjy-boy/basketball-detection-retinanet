import torch
import cv2
import numpy as np
import os
from torchvision import transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn

# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

weights_path = 'weights/best_model.pth'          # 模型权重
input_frames_dir = 'data/video_frames'           # 抽帧图片文件夹
output_dir = 'results/detected_frames_filtered'  # 筛选结果保存文件夹
os.makedirs(output_dir, exist_ok=True)

conf_threshold = 0.1          # 检测置信度阈值（可调）
max_boxes_per_image = 3        # 只保留检测框数量 ≤ 3 的帧
# ==============================================

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

transform = T.Compose([
    T.ToTensor(),
    T.Resize((640, 640)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 获取所有图片文件
image_files = [f for f in os.listdir(input_frames_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()
print(f"共找到 {len(image_files)} 张图片")

saved_count = 0
for idx, img_file in enumerate(image_files):
    img_path = os.path.join(input_frames_dir, img_file)
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    mask = (predictions['labels'] == 1) & (predictions['scores'] > conf_threshold)
    boxes = predictions['boxes'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()

    # 筛选：检测框数量在 1 到 max_boxes_per_image 之间
    if 1 <= len(boxes) <= max_boxes_per_image:
        # 在原图上绘制检测框
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        out_path = os.path.join(output_dir, f"detected_{img_file}")
        cv2.imwrite(out_path, frame)
        saved_count += 1
        print(f"已保存: {out_path} 框数={len(boxes)}")

    if idx % 100 == 0:
        print(f"处理进度: {idx}/{len(image_files)}")

print(f"\n筛选完成！共保存 {saved_count} 张图片到 {output_dir}")