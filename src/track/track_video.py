"""
篮球视频跟踪脚本（ExpC 最优模型 + SORT 跟踪）
功能：逐帧检测篮球 + SORT 关联 + 视频输出 + 检测结果 JSON
改进：统一模型加载、权重兼容、帧率统计、置信度显示、漏检标记、跟踪统计
"""

import torch
import cv2
import numpy as np
import json
import time
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision import transforms as T
from sort import Sort
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 配置 ====================
WEIGHTS_PATH = 'weights/expC_50_epoch50.pth'
IMAGE_SIZE = 800
CONF_THRESHOLD = 0.3
SORT_MAX_AGE = 5
SORT_MIN_HITS = 1

# 误检过滤参数
NMS_THRESHOLD = 0.5
MIN_BOX_SIZE = 15
MAX_BOX_SIZE = 500
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
# ================================================


def nms(boxes, scores, iou_threshold):
    """非极大值抑制，去除重叠框"""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def filter_boxes(boxes, scores):
    """NMS + 尺寸过滤 + 宽高比过滤"""
    if len(boxes) == 0:
        return boxes, scores

    keep = nms(boxes, scores, NMS_THRESHOLD)
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return boxes, scores

    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    size_mask = (bw >= MIN_BOX_SIZE) & (bh >= MIN_BOX_SIZE) & \
                (bw <= MAX_BOX_SIZE) & (bh <= MAX_BOX_SIZE)

    aspect = bw / (bh + 1e-6)
    ratio_mask = (aspect >= MIN_ASPECT_RATIO) & (aspect <= MAX_ASPECT_RATIO)

    valid = size_mask & ratio_mask
    return boxes[valid], scores[valid]


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


model = load_model(WEIGHTS_PATH, device)

transform = T.Compose([
    T.ToTensor(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def track_video(video_path, output_path, det_json_path, conf_threshold=CONF_THRESHOLD):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS)
    frame_count = 0
    video_results = {}

    # 统计
    total_detections = 0
    miss_frames = 0
    id_switches = 0
    prev_track_id = None
    start_time = time.time()

    print(f"\n开始处理: {video_path}")
    print(f"  分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    print(f"  置信度阈值: {conf_threshold}, SORT: max_age={SORT_MAX_AGE}, min_hits={SORT_MIN_HITS}")
    print(f"  输出视频: {output_path}")
    print(f"  检测JSON: {det_json_path}")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(img_tensor)[0]

        # 筛选篮球（类别1）
        mask = (predictions['labels'] == 1) & (predictions['scores'] > conf_threshold)
        boxes = predictions['boxes'][mask].cpu().numpy()
        scores = predictions['scores'][mask].cpu().numpy()

        # 坐标缩放回原图
        orig_h, orig_w = frame.shape[:2]
        sx = orig_w / IMAGE_SIZE
        sy = orig_h / IMAGE_SIZE
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        # 误检过滤：NMS + 尺寸 + 宽高比
        boxes, scores = filter_boxes(boxes, scores)

        dets = np.column_stack((boxes, scores)) if len(boxes) > 0 else np.empty((0, 5))
        tracked_objects = tracker.update(dets)

        # 保存检测结果
        frame_name = f"frame_{frame_count:04d}"
        video_results[frame_name] = []
        current_track_id = None

        if len(tracked_objects) > 0:
            total_detections += len(tracked_objects)
            for i, obj in enumerate(tracked_objects):
                x1, y1, x2, y2, track_id = obj
                track_id = int(track_id)
                current_track_id = track_id
                score = float(scores[i]) if i < len(scores) else 0.0

                video_results[frame_name].append({
                    'x1': float(x1), 'y1': float(y1),
                    'x2': float(x2), 'y2': float(y2),
                    'track_id': track_id,
                    'score': score
                })

                # 绘制检测框（绿色）
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 显示 ID 和置信度
                cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"conf:{score:.2f}",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            miss_frames += 1
            cv2.putText(frame, "MISS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ID 切换检测（单目标场景）
        if prev_track_id is not None and current_track_id is not None and prev_track_id != current_track_id:
            id_switches += 1
        prev_track_id = current_track_id

        # 帧号显示
        cv2.putText(frame, f"Frame: {frame_count}", (width - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame)
        frame_count += 1

        # 进度显示
        if frame_count % 50 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  进度: {frame_count}/{total_frames} | "
                  f"速度: {current_fps:.1f} FPS | "
                  f"检测: {total_detections} | "
                  f"漏检: {miss_frames}")

    cap.release()
    out.release()

    # 保存检测结果JSON
    with open(det_json_path, 'w', encoding='utf-8') as f:
        json.dump(video_results, f, indent=2, ensure_ascii=False)

    # 汇总统计
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    miss_rate = miss_frames / frame_count * 100 if frame_count > 0 else 0

    print("-" * 50)
    print(f"处理完成！")
    print(f"  总帧数: {frame_count}")
    print(f"  处理速度: {avg_fps:.1f} FPS（耗时 {elapsed:.1f}s）")
    print(f"  总检测数: {total_detections}")
    print(f"  漏检帧数: {miss_frames} ({miss_rate:.1f}%)")
    print(f"  ID切换次数: {id_switches}")
    print(f"  视频保存至: {output_path}")
    print(f"  检测结果保存至: {det_json_path}")


if __name__ == '__main__':
    input_video = 'data/videos/test_video.mp4'
    output_video = 'results/tracked_output.mp4'
    det_json_path = 'results/video_detections.json'
    os.makedirs('results', exist_ok=True)
    track_video(input_video, output_video, det_json_path)
