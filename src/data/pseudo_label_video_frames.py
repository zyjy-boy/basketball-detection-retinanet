import torch
import cv2
import numpy as np
import os
from torchvision import transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn

# ==================== 配置参数 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 训练好的模型权重路径
weights_path = 'weights/best_model.pth'   # 你可以换成其他权重，如 epoch80.pth

# 图片文件夹路径（你的900张照片）
frames_dir = r'.\data\video_frames'

# 标签输出文件夹（会在 frames_dir 下创建 labels 文件夹）
output_labels_dir = os.path.join(frames_dir, 'labels')
os.makedirs(output_labels_dir, exist_ok=True)

# 检测置信度阈值（可根据效果调整，建议先试0.3）
conf_threshold = 0.3
# ===================================================

# 加载模型（必须与训练时结构一致）
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

# 图像预处理（与训练时一致）
transform = T.Compose([
    T.ToTensor(),
    T.Resize((640, 640)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 遍历图片
image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"共找到 {len(image_files)} 张图片")

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(frames_dir, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"警告：无法读取 {img_path}，跳过")
        continue

    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # 筛选篮球（类别1）且置信度高于阈值
    mask = (predictions['labels'] == 1) & (predictions['scores'] > conf_threshold)
    boxes = predictions['boxes'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()

    # 如果检测到多个框，只保留置信度最高的一个
    if len(boxes) > 1:
        best_idx = np.argmax(scores)
        boxes = boxes[best_idx:best_idx+1]   # 保留最高分框
        scores = scores[best_idx:best_idx+1]

    # 生成标签文件（YOLO格式）
    if len(boxes) > 0:
        txt_filename = os.path.splitext(img_file)[0] + '.txt'
        txt_path = os.path.join(output_labels_dir, txt_filename)
        with open(txt_path, 'w') as f:
            for box in boxes:
                x1, y1, x2, y2 = box
                # 转换为YOLO格式：类别  x_center  y_center  width  height
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                # 类别ID固定为0（因为你的训练数据中篮球类别在标签里是0，但模型输出是1，这里要对应）
                # 注意：你的数据集最终使用的篮球类别ID是0还是1？
                # 根据你的 basketball_dataset.py，训练时加载的YOLO标签中，类别0被映射为labels=1。
                # 因此，生成的标签中类别应该写0，因为你的数据加载器会将0转为1。
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
        print(f"已生成: {txt_path}, 框数={len(boxes)}")
    else:
        print(f"无检测: {img_file}")

    if (idx+1) % 100 == 0:
        print(f"已处理 {idx+1}/{len(image_files)} 张")

print("伪标注完成！")