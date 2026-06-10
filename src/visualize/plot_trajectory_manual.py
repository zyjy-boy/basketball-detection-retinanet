"""
手动选择帧范围，绘制 test_video 三种轨迹对比图
交互式输入帧范围，支持多段选择

用法：python plot_trajectory_manual.py
"""

import cv2
import torch
import numpy as np
import os
import json
from torchvision import transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 简化 SORT 实现 ====================
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        self.kf.statePost = np.array([[x1+w/2],[y1+h/2],[w],[h],[0],[0],[0]], np.float32)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.bbox = bbox

    def predict(self):
        pred = self.kf.predict()
        cx, cy, w, h = pred[0,0], pred[1,0], pred[2,0], pred[3,0]
        self.bbox = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
        return self.bbox

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2-x1, y2-y1
        self.kf.correct(np.array([[x1+w/2],[y1+h/2],[w],[h]], np.float32))
        self.time_since_update = 0
        self.hit_streak += 1
        self.bbox = bbox

    def get_state(self):
        return self.bbox


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            tb = trk.bbox
            x1, y1 = max(det[0],tb[0]), max(det[1],tb[1])
            x2, y2 = min(det[2],tb[2]), min(det[3],tb[3])
            iw, ih = max(0,x2-x1), max(0,y2-y1)
            inter = iw * ih
            union = (det[2]-det[0])*(det[3]-det[1]) + (tb[2]-tb[0])*(tb[3]-tb[1]) - inter
            iou_matrix[d,t] = inter/union if union > 0 else 0
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array([row_ind, col_ind]).T
    unmatched_dets = [d for d in range(len(detections)) if d not in row_ind]
    unmatched_trks = [t for t in range(len(trackers)) if t not in col_ind]
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0],m[1]] >= iou_threshold:
            matches.append(m.reshape(1,2))
        else:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
    matches = np.concatenate(matches, axis=0) if matches else np.empty((0,2), dtype=int)
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


class Sort:
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.trackers, self.frame_count = [], 0

    def update(self, dets):
        self.frame_count += 1
        for trk in self.trackers:
            trk.predict()
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, self.trackers, self.iou_threshold)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]][:4])
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i][:4]))
        ret = []
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
                continue
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                ret.append(trk.get_state() + [trk.id])
        return ret


# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = 'weights/expC_50_epoch50.pth'
IMAGE_SIZE = 800
conf_threshold = 0.1

VIDEO_DIR = r'.\data\videos'
GT_DIR = r'results/gt_annotations'
OUTPUT_DIR = r'results/trajectory_comparison'

VIDEO_CONFIG = {
    '1': {'name': 'test_video',   'path': os.path.join(VIDEO_DIR, 'test_video.mp4'),   'gt': os.path.join(GT_DIR, 'test_video.json')},
    '2': {'name': 'test_video_03', 'path': os.path.join(VIDEO_DIR, 'test_video_03.mp4'), 'gt': os.path.join(GT_DIR, 'test_video_03.json')},
    '3': {'name': 'test_video_04', 'path': os.path.join(VIDEO_DIR, 'test_video_04.mp4'), 'gt': os.path.join(GT_DIR, 'test_video_04.json')},
}
# ===================================================


def load_model(weight_path, device):
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
    return model


model = load_model(weights_path, device)

transform = T.Compose([
    T.ToTensor(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def detect_ball_box(frame):
    orig_h, orig_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    mask = (predictions['labels'] == 1) & (predictions['scores'] > conf_threshold)
    boxes = predictions['boxes'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()
    if len(boxes) == 0:
        return None, None
    best_idx = np.argmax(scores)
    box = boxes[best_idx]
    score = scores[best_idx]
    sx, sy = orig_w / IMAGE_SIZE, orig_h / IMAGE_SIZE
    x1, y1, x2, y2 = box * np.array([sx, sy, sx, sy])
    return [x1, y1, x2, y2], score


def get_center(box):
    if box is None:
        return None
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def load_gt(gt_path):
    if not os.path.exists(gt_path):
        return None
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    points = {}
    for frame_name, boxes in gt_data.items():
        if boxes and len(boxes) > 0:
            box = boxes[0]
            points[frame_name] = ((box['x1']+box['x2'])/2, (box['y1']+box['y2'])/2)
        else:
            points[frame_name] = None
    return points


def sort_tracking(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)
    points = []
    for frame_idx in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            points.append(None)
            tracker.update([])
            continue
        box, score = detect_ball_box(frame)
        dets = [[box[0], box[1], box[2], box[3], score]] if box is not None else []
        tracks = tracker.update(dets)
        if len(tracks) > 0:
            points.append(get_center(tracks[0][:4]))
        else:
            points.append(None)
    cap.release()
    return points


def static_interpolation(video_path, start, end, sample_interval=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    sampled_points = {}
    for frame_idx in range(start, end + 1, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            sampled_points[frame_idx] = None
            continue
        box, score = detect_ball_box(frame)
        sampled_points[frame_idx] = get_center(box)
    cap.release()

    points = []
    for frame_idx in range(start, end + 1):
        if frame_idx in sampled_points:
            points.append(sampled_points[frame_idx])
            continue
        prev_idx = next_idx = None
        for offset in range(1, sample_interval + 1):
            if frame_idx - offset in sampled_points and sampled_points[frame_idx - offset] is not None:
                prev_idx = frame_idx - offset
                break
        for offset in range(1, sample_interval + 1):
            if frame_idx + offset in sampled_points and sampled_points[frame_idx + offset] is not None:
                next_idx = frame_idx + offset
                break
        if prev_idx is not None and next_idx is not None:
            p1, p2 = sampled_points[prev_idx], sampled_points[next_idx]
            t = (frame_idx - prev_idx) / (next_idx - prev_idx)
            points.append((p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1])))
        elif prev_idx is not None:
            points.append(sampled_points[prev_idx])
        elif next_idx is not None:
            points.append(sampled_points[next_idx])
        else:
            points.append(None)
    return points


def plot_trajectory(gt_points, m1_points, m2_points, bg_img, output_path, title):
    h, w = bg_img.shape[:2]
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.imshow(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))

    def plot_line(points, color, label, linestyle='-'):
        xs = [p[0] for p in points if p is not None]
        ys = [p[1] for p in points if p is not None]
        if len(xs) >= 2:
            ax.plot(xs, ys, linestyle, color=color, linewidth=2.5, label=label)
        if xs:
            ax.scatter(xs, ys, c=color, s=30, zorder=5)

    plot_line(gt_points, 'red', '真实轨迹')
    plot_line(m1_points, 'blue', 'SORT跟踪')
    plot_line(m2_points, 'green', '抽帧+插值')

    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_range(video_path, video_name, start, end, gt_data, seg_label):
    """处理一个帧范围"""
    print(f"\n  处理: 帧 {start}-{end} ...")

    # SORT跟踪（全帧）
    m1_all = sort_tracking(video_path, start, end)

    # 抽帧+插值（全帧）
    m2_all = static_interpolation(video_path, start, end, 5)

    # 只取有GT的帧
    gt_points, m1_points, m2_points = [], [], []
    for idx in range(start, end + 1):
        fn = f'frame_{idx:04d}'
        gt_pt = gt_data.get(fn)
        if gt_pt is not None:
            gt_points.append(gt_pt)
            m1_points.append(m1_all[idx - start])
            m2_points.append(m2_all[idx - start])

    print(f"    GT帧数: {len(gt_points)}")

    # 背景图
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, bg_img = cap.read()
    cap.release()
    if bg_img is None:
        bg_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

    # 画图
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUTPUT_DIR, f'{video_name}_{seg_label}_trajectory.png')
    title = f'{video_name} 帧 {start}-{end} (GT {len(gt_points)}帧)'
    plot_trajectory(gt_points, m1_points, m2_points, bg_img, fig_path, title)
    print(f"    保存至: {fig_path}")

    # 计算指标
    img_diag = np.sqrt(bg_img.shape[0]**2 + bg_img.shape[1]**2)
    total = len(gt_points)
    if total == 0:
        return
    for name, pts in [("SORT", m1_points), ("静态", m2_points)]:
        error = 0
        overlap = 0
        threshold = img_diag * 0.05
        for g, p in zip(gt_points, pts):
            if g is not None and p is not None:
                dist = np.sqrt((g[0]-p[0])**2 + (g[1]-p[1])**2)
                error += dist / img_diag
                if dist <= threshold:
                    overlap += 1
            elif g is not None and p is None:
                error += 1.0
        print(f"    {name}: NTE={error/total:.4f}, Overlap@5%={overlap/total:.4f}")


def main():
    print("=" * 60)
    print("手动选择帧范围 - 轨迹对比图")
    print("=" * 60)

    # 选择视频
    print(f"\n可选视频:")
    for key, cfg in VIDEO_CONFIG.items():
        print(f"  {key}. {cfg['name']}")
    print(f"  q. 退出")

    choice = input(f"\n选择视频 (1/2/3): ").strip()
    if choice.lower() == 'q' or choice not in VIDEO_CONFIG:
        print("退出")
        return

    cfg = VIDEO_CONFIG[choice]
    video_name = cfg['name']
    video_path = cfg['path']
    gt_path = cfg['gt']

    print(f"\n当前视频: {video_name}")

    gt_data = load_gt(gt_path)
    if gt_data is None:
        print(f"✗ GT文件不存在: {gt_path}")
        return

    # 显示GT帧范围
    valid_indices = sorted([int(fn.replace('frame_', '')) for fn, pt in gt_data.items() if pt is not None])
    if not valid_indices:
        print("✗ GT中没有篮球帧")
        return

    print(f"GT有篮球帧范围: {valid_indices[0]} - {valid_indices[-1]}")
    print(f"共 {len(valid_indices)} 个GT帧")
    print(f"\n输入格式: 起始帧 结束帧 (例如: 270 375)")
    print(f"输入 'all' 使用全部GT帧")
    print(f"输入 'q' 退出\n")

    seg_count = 1
    while True:
        user_input = input(f"段{seg_count} 帧范围 (起始 结束): ").strip()

        if user_input.lower() == 'q':
            break

        if user_input.lower() == 'all':
            start = valid_indices[0]
            end = valid_indices[-1]
        else:
            parts = user_input.split()
            if len(parts) != 2:
                print("  格式错误，请输入: 起始帧 结束帧")
                continue
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                print("  请输入数字")
                continue

        if start > end:
            start, end = end, start

        # 检查该范围内是否有GT帧
        gt_in_range = [i for i in valid_indices if start <= i <= end]
        if len(gt_in_range) == 0:
            print(f"  帧 {start}-{end} 内没有GT帧，请重新输入")
            continue

        print(f"  范围内有 {len(gt_in_range)} 个GT帧")

        process_range(video_path, video_name, start, end, gt_data, f'manual_seg{seg_count}')
        seg_count += 1

    print(f"\n完成！共生成 {seg_count-1} 张轨迹图")


if __name__ == '__main__':
    main()
