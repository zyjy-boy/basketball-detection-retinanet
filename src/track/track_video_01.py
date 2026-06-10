import torch
import cv2
import numpy as np
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision import transforms as T
from sort import Sort
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============ 优化参数 ============
IMAGE_SIZE = 1280          # 提高分辨率：800 → 1280，增强小目标检测能力
CONF_THRESHOLD = 0.05      # 降低阈值：0.3 → 0.05，减少远视角漏检
# ==================================

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

model = load_model('weights/expC_50_epoch50.pth')

transform = T.Compose([
    T.ToTensor(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def track_video(video_path, output_path, conf_threshold=CONF_THRESHOLD):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = Sort(max_age=15, min_hits=1)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(img_tensor)[0]

        # 筛选篮球（类别1），置信度高于阈值
        mask = (predictions['labels'] == 1) & (predictions['scores'] > conf_threshold)
        boxes = predictions['boxes'][mask].cpu().numpy()
        scores = predictions['scores'][mask].cpu().numpy()

        # 坐标从 IMAGE_SIZE 缩放回原图尺寸
        orig_h, orig_w = frame.shape[:2]
        sx = orig_w / IMAGE_SIZE
        sy = orig_h / IMAGE_SIZE
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        # 调试输出：每10帧打印一次
        if frame_count % 10 == 0:
            print(f"帧 {frame_count}: 检测到 {len(boxes)} 个篮球")

        dets = np.column_stack((boxes, scores)) if len(boxes) > 0 else np.empty((0, 5))
        tracked_objects = tracker.update(dets)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"视频处理完成！共 {frame_count} 帧，输出保存至 {output_path}")

if __name__ == '__main__':
    input_video = 'data/videos/test_video_01.mp4'
    output_video = 'results/tracked_output_05.mp4'
    os.makedirs('results', exist_ok=True)
    track_video(input_video, output_video, conf_threshold=CONF_THRESHOLD)
