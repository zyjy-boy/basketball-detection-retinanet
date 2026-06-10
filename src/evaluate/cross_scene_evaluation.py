import os
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn
from tqdm import tqdm
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型权重路径（可尝试使用最后epoch的权重，如 'weights/retinanet_epoch130.pth'）
weights_path = 'weights/best_model.pth'  # 或改用 'weights/retinanet_epoch130.pth'

# 四个场景的图片文件夹路径
scene_dirs = {
    '近景': r'.\data\selected_images\near',
    '远景': r'.\data\selected_images\far',
    '明亮': r'.\data\selected_images\brigh',
    '暗光': r'.\data\selected_images\dark'
}

# 输出目录
output_img_dir = 'results/cross_scene_images'
os.makedirs(output_img_dir, exist_ok=True)

# 优化后的检测参数
conf_threshold = 0.4        # 降低阈值，提高召回率（原0.55过高）
nms_iou_threshold = 0.5      # NMS IoU阈值（保持0.5）
image_size = 640             # 输入尺寸与训练一致
# ================================================

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

# 图像预处理（保持与训练一致）
transform = T.Compose([
    T.ToTensor(),
    T.Resize((image_size, image_size)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_scene(scene_name, img_dir):
    """对一个场景进行评估，返回mAP和召回率，并保存带框图片"""
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"场景 {scene_name} 共有 {len(img_files)} 张图片")

    preds = []
    targets = []
    label_dir = os.path.join(img_dir, 'labels')

    for img_file in tqdm(img_files, desc=f"处理 {scene_name}"):
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        # 读取真实框
        base = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, base + '.txt')
        if not os.path.exists(label_path):
            continue

        gt_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, bw, bh = map(float, parts)
                    # 强制类别为1（与模型输出一致）
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        gt_boxes.append([x1, y1, x2, y2])

        if not gt_boxes:
            continue

        # 模型预测
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = transform(frame_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)[0]

        # 筛选篮球（类别1）且置信度 > conf_threshold
        mask = (pred['labels'] == 1) & (pred['scores'] > conf_threshold)
        boxes = pred['boxes'][mask].cpu().numpy()
        scores = pred['scores'][mask].cpu().numpy()

        # 应用NMS去重（可选，但torchmetrics会自动处理，这里可省略）
        # 直接使用所有框

        preds.append({
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'scores': torch.tensor(scores, dtype=torch.float32),
            'labels': torch.ones(len(boxes), dtype=torch.int64)
        })
        targets.append({
            'boxes': torch.tensor(gt_boxes, dtype=torch.float32),
            'labels': torch.ones(len(gt_boxes), dtype=torch.int64)
        })

        # 保存带框图片（定性分析）
        img_with_boxes = image.copy()
        for box in gt_boxes:
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(img_with_boxes, (x1,y1), (x2,y2), (0,0,255), 2)
        for box, score in zip(boxes, scores):
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(img_with_boxes, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img_with_boxes, f'{score:.2f}', (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out_path = os.path.join(output_img_dir, f"{scene_name}_{img_file}")
        cv2.imwrite(out_path, img_with_boxes)

    # 计算指标
    if not preds or not targets:
        print(f"场景 {scene_name} 无有效数据，返回 -1")
        return -1.0, -1.0, -1.0, len(img_files)

    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=False)
    metric.update(preds, targets)
    results = metric.compute()
    mAP = results['map_50'].item()
    recall = results['mar_100'].item() if 'mar_100' in results else 0.0
    recall_small = results.get('mar_small', 0.0).item()
    return mAP, recall, recall_small, len(img_files)

def main():
    results = []
    for scene_name, img_dir in scene_dirs.items():
        if not os.path.exists(img_dir):
            print(f"场景目录 {img_dir} 不存在，跳过")
            continue
        print(f"\n评估场景: {scene_name}")
        mAP, recall, recall_small, num_imgs = evaluate_scene(scene_name, img_dir)
        results.append({
            '场景': scene_name,
            '图片数量': num_imgs,
            'mAP@0.5': f"{mAP:.4f}",
            '召回率 (mar@100)': f"{recall:.4f}",
            '小目标召回率': f"{recall_small:.4f}"
        })

    df = pd.DataFrame(results)
    print("\n跨场景评估结果:")
    print(df.to_string(index=False))
    df.to_csv('results/cross_scene_results.csv', index=False)
    print(f"\n带框图片已保存至 {output_img_dir}")
    print(f"评估表格已保存至 results/cross_scene_results.csv")

if __name__ == '__main__':
    main()