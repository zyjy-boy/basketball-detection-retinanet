"""
原始消融实验统一评估脚本（四组模型 × 合并测试集223张）—— 修正版

修正内容：
  1. mAP 计算：实现基于 PR 曲线积分的 AP（11-point interpolation）
  2. 预测框排序：同步重排 pred_boxes / pred_scores / iou_matrix
  3. parse_orig_label：移除误导性参数，统一缩放逻辑

实验设计（原始）：
  Baseline = 小数据集 + 基础策略
  ExpA     = 大数据集(1956张) + 基础策略
  ExpB     = 小数据集 + 优化策略
  ExpC     = 大数据集(1956张) + 优化策略
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import box_iou
from PIL import Image


# =================== 配置 ===================
# 四组实验的模型权重（请根据实际路径修改）
MODELS = {
    'Baseline(小数据+基础)': r'weights\baseline_final.pth',
    'ExpA(大数据+基础)': r'weights\expA_epoch50.pth',
    'ExpB(小数据+优化)': r'weights\expB_epoch50.pth',
    'ExpC(大数据+优化)': r'weights\expC_50_epoch50.pth',
}

# 原有验证集（list_file 格式，所有框都是篮球）
ORIG_LIST = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\test.txt'

# Tracer 测试集（YOLO 目录格式，类别0=篮球）
TRACER_IMG_DIR = r'.\data\SportsMOT\Tracer-basketball.v3i.yolov8\test\images'
TRACER_LBL_DIR = r'.\data\SportsMOT\Tracer-basketball.v3i.yolov8\test\labels'
TRACER_BBALL_CID = 0  # 类别0=篮球

# 评估参数
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
IMAGE_SIZE = 640
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ==========================================


def load_model(weight_path, device):
    """加载 RetinaNet 模型（兼容多种权重格式）"""
    print(f"\n正在加载模型: {weight_path}")
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
    print(f"  ✓ 模型加载成功")
    return model


def load_image(img_path):
    """加载并预处理图像"""
    image = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image.size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
    return img_tensor, orig_w, orig_h


def parse_yolo_boxes(label_path, class_id, orig_w, orig_h):
    """解析 YOLO 标签，只保留指定类别，返回 IMAGE_SIZE 尺度的绝对坐标"""
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


def parse_orig_label(label_path):
    """解析原有数据集的标签文件（每行: class_id x_center y_center width height，归一化坐标）

    归一化坐标直接乘以 IMAGE_SIZE 即可得到模型推理尺度下的绝对坐标，
    无需原图尺寸信息。
    """
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
                    if cid != 0:  # 只取篮球（类别0）
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


def build_combined_test_set():
    """合并两个测试集，返回 [(img_path, label_path, source_type), ...]
    source_type: 'orig' 或 'tracer'
    """
    items = []

    # 1. 原有验证集（list_file 格式）
    print(f"\n加载原有验证集: {ORIG_LIST}")
    count_orig = 0
    with open(ORIG_LIST, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and os.path.exists(parts[0]):
                items.append((parts[0], parts[1], 'orig'))
                count_orig += 1
    print(f"  ✓ 原有验证集: {count_orig} 张")

    # 2. Tracer 测试集（YOLO 目录格式）
    print(f"\n加载 Tracer 测试集: {TRACER_IMG_DIR}")
    count_tracer = 0
    for fname in sorted(os.listdir(TRACER_IMG_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(fname)[0]
            img_path = os.path.join(TRACER_IMG_DIR, fname)
            lbl_path = os.path.join(TRACER_LBL_DIR, name + '.txt')
            if os.path.exists(lbl_path):
                items.append((img_path, lbl_path, 'tracer'))
                count_tracer += 1
    print(f"  ✓ Tracer 测试集: {count_tracer} 张")

    print(f"\n合并测试集总计: {len(items)} 张 ({count_orig} + {count_tracer})")
    return items


def get_gt_boxes(label_path, source_type, orig_w, orig_h):
    """根据来源类型获取真实框"""
    if source_type == 'orig':
        return parse_orig_label(label_path)
    else:  # tracer
        return parse_yolo_boxes(label_path, TRACER_BBALL_CID, orig_w, orig_h)


def compute_ap(all_scores, all_tp_flags, num_gt):
    """基于 11-point interpolation 计算 AP（Average Precision）

    Args:
        all_scores:  所有预测框的置信度列表（float）
        all_tp_flags: 所有预测框的 TP/FP 标志列表（1=TP, 0=FP），与 all_scores 一一对应
        num_gt:      真实框总数

    Returns:
        ap: Average Precision
    """
    if num_gt == 0 or len(all_scores) == 0:
        return 0.0

    # 按置信度降序排序
    sorted_indices = np.argsort(-np.array(all_scores))
    tp_sorted = np.array(all_tp_flags)[sorted_indices].astype(float)

    # 累计 TP 和 FP
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)

    # 计算 Precision 和 Recall
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / num_gt

    # 11-point interpolation AP
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        # 找到 recall >= t 时最大的 precision
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    ap /= 11.0

    return ap


def evaluate(model, items, device):
    """在合并测试集上评估模型

    双路径评估：
      - AP 计算：置信度阈值 = 0.0，所有预测框参与 PR 曲线积分
      - P/R/F1：使用原始 CONF_THRESHOLD 过滤，展示固定工作点性能
    """
    # 固定阈值下的统计（用于 P/R/F1）
    total_tp_thr, total_fp_thr, total_gt = 0, 0, 0

    # AP 计算所需：收集所有预测框（conf >= 0.0）的置信度和 TP/FP 标志
    all_scores = []
    all_tp_flags = []

    for img_path, label_path, source_type in tqdm(items, desc="评估中"):
        if not os.path.exists(img_path):
            continue

        img_tensor, orig_w, orig_h = load_image(img_path)
        gt_boxes = get_gt_boxes(label_path, source_type, orig_w, orig_h)

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
                if best_iou >= IOU_THRESHOLD and best_gt.item() not in matched:
                    is_tp = True
                    matched.add(best_gt.item())

            score = pred_scores[i].item()

            # 所有框都参与 AP 计算
            all_scores.append(score)
            all_tp_flags.append(1 if is_tp else 0)

            # 仅 conf >= CONF_THRESHOLD 的框参与固定阈值 P/R/F1
            if score >= CONF_THRESHOLD:
                if is_tp:
                    total_tp_thr += 1
                else:
                    total_fp_thr += 1

    if total_gt == 0:
        return None

    # 固定阈值下的 P/R/F1
    p = total_tp_thr / (total_tp_thr + total_fp_thr) if (total_tp_thr + total_fp_thr) > 0 else 0
    r = total_tp_thr / total_gt
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    # AP：基于所有预测框的 PR 曲线积分
    ap = compute_ap(all_scores, all_tp_flags, total_gt)

    return {
        'mAP': ap,          # 单类别 AP 即为 mAP（所有框参与）
        'precision': p,     # CONF_THRESHOLD 阈值下的 Precision
        'recall': r,        # CONF_THRESHOLD 阈值下的 Recall
        'f1': f1,           # CONF_THRESHOLD 阈值下的 F1
        'tp': total_tp_thr,
        'fp': total_fp_thr,
        'gt': total_gt
    }


def main():
    print("=" * 70)
    print("原始消融实验统一评估（四组模型 × 合并测试集）—— 修正版")
    print(f"设备: {DEVICE} | IoU阈值: {IOU_THRESHOLD}")
    print(f"  AP 计算: conf >= 0.0（所有框参与 PR 曲线积分）")
    print(f"  P/R/F1:  conf >= {CONF_THRESHOLD}（固定工作点）")
    print("=" * 70)

    # 检查权重文件
    print("\n待评估模型:")
    for name, path in MODELS.items():
        exists = "✓" if os.path.exists(path) else "✗ 不存在"
        print(f"  {name}: {path} [{exists}]")

    missing = [n for n, p in MODELS.items() if not os.path.exists(p)]
    if missing:
        print(f"\n⚠ 以下模型权重不存在:")
        for n in missing:
            print(f"  - {n}: {MODELS[n]}")
        print("请检查路径后重新运行。")
        return

    # 构建合并测试集
    items = build_combined_test_set()
    if len(items) == 0:
        print("✗ 合并测试集为空，请检查路径")
        return

    # 评估每个模型
    results = {}
    for model_name, model_path in MODELS.items():
        print(f"\n{'='*70}")
        print(f"评估: {model_name}")
        print(f"{'='*70}")

        model = load_model(model_path, DEVICE)
        result = evaluate(model, items, DEVICE)

        if result:
            results[model_name] = result
            print(f"\n  AP@0.5={result['mAP']:.4f} | P@{CONF_THRESHOLD}={result['precision']:.4f} | R@{CONF_THRESHOLD}={result['recall']:.4f} | F1@{CONF_THRESHOLD}={result['f1']:.4f}")
            print(f"  TP@{CONF_THRESHOLD}={result['tp']} | FP@{CONF_THRESHOLD}={result['fp']} | GT={result['gt']}")
        else:
            print(f"\n  ⚠ 无篮球标注，评估失败")

        del model
        torch.cuda.empty_cache()

    # 汇总表格
    print("\n" + "=" * 80)
    print("原始消融实验结果汇总（合并测试集）")
    print("=" * 80)
    print(f"{'实验':<28}| {'AP@0.5':<10}| {'P@0.3':<10}| {'R@0.3':<10}| {'F1@0.3':<10}| {'TP':<6}| {'FP':<6}| {'GT':<6}")
    print("-" * 80)
    for mn in MODELS:
        if mn not in results:
            print(f"{mn:<28}| {'加载失败':<10}")
            continue
        r = results[mn]
        print(f"{mn:<28}| {r['mAP']:<10.4f}| {r['precision']:<10.4f}| {r['recall']:<10.4f}| {r['f1']:<10.4f}| {r['tp']:<6}| {r['fp']:<6}| {r['gt']:<6}")
    print("-" * 80)

    # 保存结果
    os.makedirs('results', exist_ok=True)
    result_file = 'results/original_ablation_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("原始消融实验结果汇总（合并测试集）—— 修正版\n")
        f.write("=" * 80 + "\n\n")
        f.write("实验设计:\n")
        f.write("  Baseline = 小数据集 + 基础策略\n")
        f.write("  ExpA     = 大数据集(1956张) + 基础策略\n")
        f.write("  ExpB     = 小数据集 + 优化策略\n")
        f.write("  ExpC     = 大数据集(1956张) + 优化策略\n\n")
        f.write("修正说明:\n")
        f.write("  1. AP 基于 11-point interpolation，置信度阈值 = 0.0（所有框参与）\n")
        f.write("  2. P/R/F1 使用原始 CONF_THRESHOLD 过滤，展示固定工作点性能\n")
        f.write("  3. 预测框排序时同步重排 pred_boxes / pred_scores / iou_matrix\n")
        f.write("  4. parse_orig_label 移除误导性的 orig_w, orig_h 参数\n\n")
        f.write(f"{'实验':<28}| {'AP@0.5':<10}| {'P@0.3':<10}| {'R@0.3':<10}| {'F1@0.3':<10}\n")
        f.write("-" * 80 + "\n")
        for mn in MODELS:
            if mn not in results:
                continue
            r = results[mn]
            f.write(f"{mn:<28}| {r['mAP']:<10.4f}| {r['precision']:<10.4f}| {r['recall']:<10.4f}| {r['f1']:<10.4f}\n")
        f.write("-" * 80 + "\n")
    print(f"\n✓ 结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
