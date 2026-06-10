"""
pr_curve.py —— 绘制ExpC最优模型在合并测试集（deepsportradar + Tracer）上的PR曲线
使用11点插值法计算AP（与论文一致）

预处理和模型加载逻辑与 evaluate_original_ablation.py 完全一致
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import box_iou
from PIL import Image
from glob import glob
from tqdm import tqdm
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============ 配置 ============
WEIGHTS_PATH = 'weights/expC_50_epoch50.pth'
IMAGE_SIZE = 640

# 原有测试集（list_file格式，每行: 图片路径 标注路径）
ORIG_LIST = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\test.txt'

# Tracer 测试集（YOLO 目录格式，类别0=篮球）
TRACER_IMG_DIR = r'.\data\SportsMOT\Tracer-basketball.v3i.yolov8\test\images'
TRACER_LBL_DIR = r'.\data\SportsMOT\Tracer-basketball.v3i.yolov8\test\labels'
TRACER_BBALL_CID = 0  # 类别0=篮球

IOU_THRESHOLD = 0.5
SAVE_PATH = 'results/pr_curve_expC.png'
# ================================


def load_image(img_path):
    """加载并预处理图像（与 evaluate_original_ablation.py 完全一致）"""
    image = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image.size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
    return img_tensor, orig_w, orig_h


def load_model(weight_path, device):
    """加载RetinaNet模型（与 evaluate_original_ablation.py 完全一致）"""
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


def parse_orig_label(label_path):
    """解析原有数据集标签（与 evaluate_original_ablation.py 完全一致）"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    if cid != 0:
                        continue
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x_min = (xc - w / 2) * IMAGE_SIZE
                    y_min = (yc - h / 2) * IMAGE_SIZE
                    x_max = (xc + w / 2) * IMAGE_SIZE
                    y_max = (yc + h / 2) * IMAGE_SIZE
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
            except:
                continue
    return boxes


def parse_yolo_boxes(label_path, class_id, orig_w, orig_h):
    """解析Tracer YOLO标签（与 evaluate_original_ablation.py 完全一致）"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                cid = int(parts[0])
                if cid != class_id:
                    continue
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                sx = IMAGE_SIZE / orig_w
                sy = IMAGE_SIZE / orig_h
                x_min = (xc - w / 2) * orig_w * sx
                y_min = (yc - h / 2) * orig_h * sy
                x_max = (xc + w / 2) * orig_w * sx
                y_max = (yc + h / 2) * orig_h * sy
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
            except:
                continue
    return boxes


def build_combined_test_set():
    """合并两个测试集（与 evaluate_original_ablation.py 完全一致）"""
    items = []

    # 1. 原有验证集
    with open(ORIG_LIST, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and os.path.exists(parts[0]):
                items.append((parts[0], parts[1], 'orig'))
    print(f"[deepsportradar] {sum(1 for _, _, s in items if s == 'orig')} 张图片")

    # 2. Tracer 测试集
    for fname in sorted(os.listdir(TRACER_IMG_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(fname)[0]
            img_path = os.path.join(TRACER_IMG_DIR, fname)
            lbl_path = os.path.join(TRACER_LBL_DIR, name + '.txt')
            if os.path.exists(lbl_path):
                items.append((img_path, lbl_path, 'tracer'))
    print(f"[Tracer] {sum(1 for _, _, s in items if s == 'tracer')} 张图片")
    print(f"合并测试集: {len(items)} 张图片")

    return items


def compute_pr_curve(model, items, device, iou_threshold=IOU_THRESHOLD):
    """计算PR曲线（与 evaluate_original_ablation.py 的 evaluate 逻辑一致）"""
    all_scores = []
    all_tp_flags = []
    total_gt = 0

    model.eval()
    for img_path, label_path, source_type in tqdm(items, desc="评估中"):
        if not os.path.exists(img_path):
            continue

        img_tensor, orig_w, orig_h = load_image(img_path)

        # 获取GT框
        if source_type == 'orig':
            gt_boxes = parse_orig_label(label_path)
        else:
            gt_boxes = parse_yolo_boxes(label_path, TRACER_BBALL_CID, orig_w, orig_h)

        if len(gt_boxes) == 0:
            continue
        total_gt += len(gt_boxes)

        with torch.no_grad():
            outputs = model([img_tensor.to(device)])

        pred_boxes = outputs[0]['boxes'].cpu()
        pred_scores = outputs[0]['scores'].cpu()
        pred_labels = outputs[0]['labels'].cpu()

        # 只筛选篮球类别（类别1），不设置信度下限
        mask = pred_labels == 1
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]

        if len(pred_boxes) == 0:
            continue

        gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
        iou_matrix = box_iou(pred_boxes, gt_tensor)

        # 按置信度降序排序，同步重排
        sorted_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_idx]
        pred_scores = pred_scores[sorted_idx]
        iou_matrix = iou_matrix[sorted_idx]

        matched = set()
        for i in range(len(pred_boxes)):
            is_tp = False
            if iou_matrix.shape[1] > 0:
                best_iou, best_gt = iou_matrix[i].max(dim=0)
                if best_iou >= iou_threshold and best_gt.item() not in matched:
                    is_tp = True
                    matched.add(best_gt.item())

            all_scores.append(pred_scores[i].item())
            all_tp_flags.append(1 if is_tp else 0)

    if total_gt == 0 or len(all_scores) == 0:
        print("警告：无有效检测结果或GT")
        return np.array([0]), np.array([0]), 0.0

    # 按置信度排序
    sorted_indices = np.argsort(-np.array(all_scores))
    all_scores = np.array(all_scores)[sorted_indices]
    all_tp_flags = np.array(all_tp_flags)[sorted_indices]

    # 计算累积TP和FP
    tp_cumsum = np.cumsum(all_tp_flags)
    fp_cumsum = np.cumsum(1 - all_tp_flags)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt

    # 11点插值法AP
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if mask.any():
            ap += np.max(precisions[mask])
    ap /= 11.0

    return precisions, recalls, ap


def plot_pr_curve(precisions, recalls, ap, save_path=SAVE_PATH):
    """绘制PR曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2, label=f'AP = {ap:.4f}')
    plt.fill_between(recalls, precisions, alpha=0.2, color='blue')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve (IoU=0.5)', fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR曲线已保存至 {save_path}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {WEIGHTS_PATH}")
    model = load_model(WEIGHTS_PATH, device)

    # 构建合并测试集
    items = build_combined_test_set()

    # 计算PR曲线
    precisions, recalls, ap = compute_pr_curve(model, items, device)
    print(f"AP (11点插值): {ap:.4f}")

    # 绘制
    plot_pr_curve(precisions, recalls, ap)
