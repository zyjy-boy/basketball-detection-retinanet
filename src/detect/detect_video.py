import torch
import cv2
import numpy as np
import os
from torchvision import transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn

# ==================== 配置参数 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型权重路径
weights_path = 'weights/best_model.pth'

# 输入视频路径（请修改为你的新视频路径）
input_video = r'.\data\videos\test_video.mp4'

# 输出视频路径
output_video = 'results/01_detected_video_low_thresh.mp4'

# 检测置信度阈值（可调）
conf_threshold = 0.6
# ===================================================

# 加载模型（与训练时一致）
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

print("加载模型中...")
model = load_model(weights_path)

transform = T.Compose([
    T.ToTensor(),
    T.Resize((800, 800)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {input_video}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
print("开始处理视频...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # 筛选篮球（类别1）且置信度高于阈值
    mask = (predictions['labels'] == 1) & (predictions['scores'] > conf_threshold)
    boxes = predictions['boxes'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()

    # 在输出视频中绘制检测框
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{score:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"已处理 {frame_count} 帧")

cap.release()
out.release()
print(f"处理完成！共 {frame_count} 帧，输出视频保存至 {output_video}")