import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from src.datasets.basketball_dataset import BasketballDataset
from torch.utils.data import DataLoader
import numpy as np

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_list = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\test.txt'
    dataset = BasketballDataset(val_list, image_size=512, transform=None)  # 不使用数据增强

    # 随机选一张图
    idx = random.randint(0, len(dataset)-1)
    img, target = dataset[idx]
    # img是tensor [C,H,W]，需转为numpy用于显示
    img_np = img.cpu().permute(1,2,0).numpy()
    # 反归一化（如果之前做了归一化）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    # 加载模型
    model = retinanet_resnet50_fpn(weights=None)
    num_classes = 2
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.cls_logits.in_channels
    model.head.classification_head.cls_logits = torch.nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
    model.head.classification_head.num_classes = num_classes
    model.load_state_dict(torch.load('weights/retinanet_epoch20.pth', map_location='cpu'))
    model.to(device)
    model.eval()

    # 预测
    img_input = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_input)[0]

    # 筛选预测（置信度>0.1）
    mask = (pred['labels'] == 1) & (pred['scores'] > 0.1)
    pred_boxes = pred['boxes'][mask].cpu().numpy()

    # 真实框
    gt_boxes = target['boxes'].cpu().numpy()

    # 画图
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.imshow(img_np)
    # 画真实框（红色）
    for box in gt_boxes:
        x1,y1,x2,y2 = box
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # 画预测框（绿色）
    for box in pred_boxes:
        x1,y1,x2,y2 = box
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.title(f"Red: GT, Green: Pred (score>{0.1})")
    plt.show()

if __name__ == '__main__':
    visualize()