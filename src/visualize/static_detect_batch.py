"""
脚本2：批量静态检测
功能：对脚本1抽取的帧图片，使用ExpC最优模型进行静态检测，输出中心点坐标
输入：data/frames/<视频名>/frame_XXXX.jpg
输出：results/static_detections/<视频名>.json
格式：{"frame_0000": {"cx": 100.5, "cy": 200.3, "conf": 0.95}, ...}
"""

import torch
import cv2
import numpy as np
import json
import os
import glob
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision import transforms as T

# ==================== 配置 ====================
WEIGHTS_PATH = 'weights/expC_50_epoch50.pth'
IMAGE_SIZE = 800
CONF_THRESHOLD = 0.3  # 静态检测用低阈值，保留更多结果

FRAMES_DIR = r'.\data\frames'
OUTPUT_DIR = r'results/static_detections'
VIDEO_NAMES = ['test_video', 'test_video_03', 'test_video_04']
# ================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def load_model(weight_path, device):
    """加载RetinaNet模型（兼容多种权重格式）"""
    print(f"加载模型: {weight_path}")
    model = retinanet_resnet50_fpn(weights=None, num_classes=2)
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("  模型加载成功")
    return model


transform = T.Compose([
    T.ToTensor(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def detect_frame(model, img_path):
    """对单张图片检测篮球，返回中心点和置信度"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    mask = (predictions['labels'] == 1) & (predictions['scores'] > CONF_THRESHOLD)
    boxes = predictions['boxes'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()

    if len(boxes) == 0:
        return None

    # 取置信度最高的
    best_idx = np.argmax(scores)
    box = boxes[best_idx]
    score = scores[best_idx]

    # 坐标缩放回原图
    sx = orig_w / IMAGE_SIZE
    sy = orig_h / IMAGE_SIZE
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 * sx
    cy = (y1 + y2) / 2 * sy
    bw = (x2 - x1) * sx
    bh = (y2 - y1) * sy

    return {"cx": float(cx), "cy": float(cy), "bw": float(bw), "bh": float(bh), "conf": float(score)}


def process_video(model, video_name):
    """处理单个视频的所有帧"""
    frame_dir = os.path.join(FRAMES_DIR, video_name)
    if not os.path.exists(frame_dir):
        print(f"  ✗ 帧目录不存在: {frame_dir}")
        return

    frame_files = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.jpg')))
    if not frame_files:
        print(f"  ✗ 未找到帧图片: {frame_dir}")
        return

    results = {}
    detected = 0
    missed = 0

    for img_path in frame_files:
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        result = detect_frame(model, img_path)
        results[frame_name] = result
        if result is not None:
            detected += 1
        else:
            missed += 1

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{video_name}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {video_name}: {len(frame_files)}帧, 检测{detected}帧, 漏检{missed}帧")
    print(f"    保存至: {output_path}")


def main():
    print("=" * 60)
    print("批量静态检测（ExpC 模型）")
    print(f"帧目录: {FRAMES_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"置信度阈值: {CONF_THRESHOLD}")
    print("=" * 60)

    model = load_model(WEIGHTS_PATH, device)

    for video_name in VIDEO_NAMES:
        print(f"\n处理: {video_name}")
        process_video(model, video_name)

    print(f"\n{'=' * 60}")
    print("完成！")


if __name__ == '__main__':
    main()
