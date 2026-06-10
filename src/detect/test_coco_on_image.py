import torch
import cv2
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
from torchvision import transforms as T
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT).to(device)
model.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Resize((640, 640)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 替换为你的验证集图片路径
img_path = r'.\data\SportsMOT\deepsportradar-DatasetNinja\test\img\1D90JA9CAH4Z_jpg.rf.18d9f413fd3642caf1b83d30557c8ab3.jpg'
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_tensor = transform(image_rgb).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img_tensor)[0]

mask = (pred['labels'] == 37) & (pred['scores'] > 0.1)
boxes = pred['boxes'][mask].cpu().numpy()
scores = pred['scores'][mask].cpu().numpy()
print(f"检测到 {len(boxes)} 个球类，置信度: {scores}")

# 画框
for box in boxes:
    x1,y1,x2,y2 = box.astype(int)
    cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
cv2.imshow('COCO detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()