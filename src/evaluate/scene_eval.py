"""
场景分类评估脚本
使用 ExpC 模型对 selected_images 下的 4 个场景子集分别评估
输出每个场景的 AP@0.5、漏检率、误检率

场景目录结构：
  selected_images/
    brigh/  — 明亮场景（含 labels/ 子目录）
    dark/   — 暗光场景
    near/   — 近景场景
    far/    — 远景场景
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import box_iou
from PIL import Image


# =================== 配置 ===================
MODEL_PATH = r'weights\expC_50_epoch50.pth'

SELECTED_ROOT = r'.\data\selected_images'

# 场景目录名 → 中文显示名
SCENES = {
    'brigh': '明亮',
    'dark':  '暗光',
    'near':  '近景',
    'far':   '远景',
}

# 评估参数
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
IMAGE_SIZE = 640
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ==========================================


def load_model(weight_path, device):
    """加载 RetinaNet 模型"""
    print(f"正在加载模型: {weight_path}")
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
    print("  ✓ 加载成功")
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


def parse_yolo_label(label_path, class_id=0):
    """解析 YOLO 标签文件，返回 IMAGE_SIZE 尺度的绝对坐标框列表"""
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
                x_min = (xc - w / 2) * IMAGE_SIZE
                y_min = (yc - h / 2) * IMAGE_SIZE
                x_max = (xc + w / 2) * IMAGE_SIZE
                y_max = (yc + h / 2) * IMAGE_SIZE
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
            except:
                continue
    return boxes


def scan_scene(scene_dir):
    """扫描场景目录，返回 [(img_path, label_path), ...]

    支持两种目录结构：
      1) labels/ 子目录：scene_dir/images/*.jpg + scene_dir/labels/*.txt
      2) 同级目录：scene_dir/*.jpg + scene_dir/*.txt
    """
    items = []

    # 优先检查 labels/ 子目录结构
    labels_subdir = os.path.join(scene_dir, 'labels')
    if os.path.isdir(labels_subdir):
        # 在场景根目录下找图片
        for fname in sorted(os.listdir(scene_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(fname)[0]
                img_path = os.path.join(scene_dir, fname)
                lbl_path = os.path.join(labels_subdir, name + '.txt')
                if os.path.exists(lbl_path):
                    items.append((img_path, lbl_path))
        return items

    # 回退：同级目录（图片和标签在同一目录下）
    for fname in sorted(os.listdir(scene_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(fname)[0]
            img_path = os.path.join(scene_dir, fname)
            lbl_path = os.path.join(scene_dir, name + '.txt')
            if os.path.exists(lbl_path):
                items.append((img_path, lbl_path))

    return items


def compute_ap(all_scores, all_tp_flags, num_gt):
    """基于 11-point interpolation 计算 AP"""
    if num_gt == 0 or len(all_scores) == 0:
        return 0.0

    sorted_indices = np.argsort(-np.array(all_scores))
    tp_sorted = np.array(all_tp_flags)[sorted_indices].astype(float)

    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)

    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / num_gt

    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    ap /= 11.0

    return ap


def evaluate_scene(model, items, device):
    """对单个场景的图片列表进行评估

    Returns:
        dict: AP@0.5, Precision, Recall, F1, 漏检率, 误检率, TP, FP, GT, 图片数
    """
    total_tp_thr, total_fp_thr, total_gt = 0, 0, 0
    all_scores = []
    all_tp_flags = []

    for img_path, label_path in tqdm(items, desc="  评估中", leave=False):
        if not os.path.exists(img_path):
            continue

        img_tensor, orig_w, orig_h = load_image(img_path)
        gt_boxes = parse_yolo_label(label_path, class_id=0)

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

            # 所有框参与 AP 计算
            all_scores.append(score)
            all_tp_flags.append(1 if is_tp else 0)

            # 仅 conf >= CONF_THRESHOLD 的框参与固定阈值指标
            if score >= CONF_THRESHOLD:
                if is_tp:
                    total_tp_thr += 1
                else:
                    total_fp_thr += 1

    if total_gt == 0:
        return None

    p = total_tp_thr / (total_tp_thr + total_fp_thr) if (total_tp_thr + total_fp_thr) > 0 else 0
    r = total_tp_thr / total_gt
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    ap = compute_ap(all_scores, all_tp_flags, total_gt)

    # 漏检率 = 未检测到的 GT / 总 GT = 1 - Recall
    miss_rate = 1.0 - r

    # 误检率 = FP / (TP + FP) = 1 - Precision（在检测任务中常用此定义）
    false_detect_rate = 1.0 - p if (total_tp_thr + total_fp_thr) > 0 else 0.0

    return {
        'ap': ap,
        'precision': p,
        'recall': r,
        'f1': f1,
        'miss_rate': miss_rate,
        'false_detect_rate': false_detect_rate,
        'tp': total_tp_thr,
        'fp': total_fp_thr,
        'gt': total_gt,
        'num_images': len(items),
    }


def main():
    print("=" * 90)
    print("场景分类评估（ExpC 模型 × 4 场景）")
    print(f"模型: {MODEL_PATH}")
    print(f"场景根目录: {SELECTED_ROOT}")
    print(f"设备: {DEVICE} | IoU阈值: {IOU_THRESHOLD} | 置信度阈值: {CONF_THRESHOLD}")
    print("=" * 90)

    if not os.path.exists(MODEL_PATH):
        print(f"✗ 模型权重不存在: {MODEL_PATH}")
        return

    if not os.path.exists(SELECTED_ROOT):
        print(f"✗ 场景目录不存在: {SELECTED_ROOT}")
        return

    # 加载模型
    model = load_model(MODEL_PATH, DEVICE)

    # 扫描各场景
    print(f"\n扫描场景目录...")
    scene_items = {}
    for scene_key, scene_name in SCENES.items():
        scene_dir = os.path.join(SELECTED_ROOT, scene_key)
        if not os.path.isdir(scene_dir):
            print(f"  ⚠ 场景目录不存在: {scene_dir}")
            continue
        items = scan_scene(scene_dir)
        scene_items[scene_key] = items
        print(f"  {scene_name}({scene_key}): {len(items)} 张")

    if not scene_items:
        print("\n✗ 未找到任何场景数据，请检查目录结构")
        return

    # 下采样：使各场景图片数量与最少的场景一致
    min_count = min(len(v) for v in scene_items.values())
    print(f"\n各场景原始数量: {', '.join(f'{SCENES[k]}={len(v)}' for k, v in scene_items.items())}")
    print(f"下采样至统一数量: {min_count} 张（取最少场景的数量）")

    rng = np.random.RandomState(42)  # 固定随机种子，保证可复现
    for scene_key in scene_items:
        items = scene_items[scene_key]
        if len(items) > min_count:
            indices = rng.choice(len(items), size=min_count, replace=False)
            indices = sorted(indices)
            scene_items[scene_key] = [items[i] for i in indices]
            print(f"  {SCENES[scene_key]}: {len(items)} → {min_count}（随机抽取）")

    # 逐场景评估
    results = {}
    for scene_key, scene_name in SCENES.items():
        if scene_key not in scene_items or len(scene_items[scene_key]) == 0:
            continue

        print(f"\n{'─'*60}")
        print(f"场景: {scene_name}（{scene_key}）— {len(scene_items[scene_key])} 张图片")
        print(f"{'─'*60}")

        result = evaluate_scene(model, scene_items[scene_key], DEVICE)
        if result:
            results[scene_key] = result
            print(f"\n  AP@0.5={result['ap']:.4f} | P={result['precision']:.4f} | R={result['recall']:.4f} | F1={result['f1']:.4f}")
            print(f"  漏检率={result['miss_rate']:.4f} | 误检率={result['false_detect_rate']:.4f}")
            print(f"  TP={result['tp']} | FP={result['fp']} | GT={result['gt']}")
        else:
            print(f"\n  ⚠ 无篮球标注，跳过")

    del model
    torch.cuda.empty_cache()

    # 汇总表格
    print("\n" + "=" * 100)
    print("场景分类评估结果汇总")
    print("=" * 100)
    header = (f"{'场景':<8}| {'图片数':<6}| {'GT':<6}| {'AP@0.5':<10}| "
              f"{'P@0.3':<10}| {'R@0.3':<10}| {'F1@0.3':<10}| "
              f"{'漏检率':<10}| {'误检率':<10}| {'TP':<6}| {'FP':<6}")
    print(header)
    print("-" * 100)
    for scene_key, scene_name in SCENES.items():
        if scene_key not in results:
            print(f"{scene_name:<8}| {'—':<6}| {'—':<6}| {'无数据':<10}")
            continue
        r = results[scene_key]
        print(f"{scene_name:<8}| {r['num_images']:<6}| {r['gt']:<6}| "
              f"{r['ap']:<10.4f}| {r['precision']:<10.4f}| {r['recall']:<10.4f}| {r['f1']:<10.4f}| "
              f"{r['miss_rate']:<10.4f}| {r['false_detect_rate']:<10.4f}| "
              f"{r['tp']:<6}| {r['fp']:<6}")
    print("-" * 100)

    # 保存结果
    os.makedirs('results', exist_ok=True)
    result_file = 'results/scene_eval_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("场景分类评估结果汇总（ExpC 模型）\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"模型: {MODEL_PATH}\n")
        f.write(f"场景根目录: {SELECTED_ROOT}\n")
        f.write(f"IoU阈值: {IOU_THRESHOLD} | 置信度阈值: {CONF_THRESHOLD}\n\n")
        f.write("指标说明:\n")
        f.write("  AP@0.5    = 11-point interpolation AP（所有框参与）\n")
        f.write("  P/R/F1    = conf >= 0.3 固定工作点\n")
        f.write("  漏检率    = 1 - Recall（未被检测到的 GT 比例）\n")
        f.write("  误检率    = 1 - Precision（FP 占总预测的比例）\n\n")
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        for scene_key, scene_name in SCENES.items():
            if scene_key not in results:
                continue
            r = results[scene_key]
            f.write(f"{scene_name:<8}| {r['num_images']:<6}| {r['gt']:<6}| "
                    f"{r['ap']:<10.4f}| {r['precision']:<10.4f}| {r['recall']:<10.4f}| {r['f1']:<10.4f}| "
                    f"{r['miss_rate']:<10.4f}| {r['false_detect_rate']:<10.4f}| "
                    f"{r['tp']:<6}| {r['fp']:<6}\n")
        f.write("-" * 100 + "\n")
    print(f"\n✓ 结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
