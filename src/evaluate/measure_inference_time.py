"""
测试最优模型在 GPU 上的单帧推理时间
输出：平均推理时间(ms)、标准差、对应帧率(FPS)

用法：python measure_inference_time.py
"""

import torch
import numpy as np
import time
import os
import glob
import cv2
from torchvision import transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn

# ==================== 配置 ====================
weights_path = 'weights/expC_50_epoch50.pth'
IMAGE_SIZE = 800
WARMUP_RUNS = 50       # 预热次数（不计入统计）
TEST_RUNS = 200        # 测试次数
# ================================================


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


def main():
    print("=" * 60)
    print("模型推理时间测试")
    print("=" * 60)

    # 设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 加载模型
    print(f"\n加载模型: {weights_path}")
    model = load_model(weights_path, device)
    print("模型加载完成")

    # 图像预处理
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 使用真实测试图像
    print(f"\n加载真实测试图像...")
    test_images = []
    # 尝试多个可能的图片目录
    img_dirs = [
        r'.\data\frames\test_video',
        r'.\data\frames\test_video_03',
        r'.\data\frames\test_video_04',
    ]
    img_files = []
    for d in img_dirs:
        if os.path.exists(d):
            files = sorted(glob.glob(os.path.join(d, '*.jpg')))
            img_files.extend(files)
            if len(img_files) >= TEST_RUNS:
                break
    if len(img_files) == 0:
        print("  未找到真实图像，使用随机图像")
        np.random.seed(42)
        for _ in range(TEST_RUNS):
            img = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).unsqueeze(0).to(device)
            test_images.append(img_tensor)
    else:
        if len(img_files) < TEST_RUNS:
            img_files = (img_files * ((TEST_RUNS // len(img_files)) + 1))[:TEST_RUNS]
        for f in img_files[:TEST_RUNS]:
            img = cv2.imread(f)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img_rgb).unsqueeze(0).to(device)
                test_images.append(img_tensor)
        print(f"  加载了 {len(test_images)} 张真实图像")

    # 预热
    print(f"\n预热 {WARMUP_RUNS} 次...")
    with torch.no_grad():
        for i in range(WARMUP_RUNS):
            _ = model(test_images[i % len(test_images)])[0]
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 正式测试
    print(f"测试 {TEST_RUNS} 次...")
    times = []
    with torch.no_grad():
        for i in range(TEST_RUNS):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(test_images[i])[0]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转为毫秒

    # 统计结果
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    fps = 1000.0 / avg_time

    print(f"\n{'=' * 60}")
    print(f"推理时间测试结果")
    print(f"{'=' * 60}")
    print(f"  测试次数:     {TEST_RUNS}")
    print(f"  输入尺寸:     {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"  平均推理时间: {avg_time:.2f} ms")
    print(f"  标准差:       {std_time:.2f} ms")
    print(f"  中位数:       {median_time:.2f} ms")
    print(f"  最小值:       {min_time:.2f} ms")
    print(f"  最大值:       {max_time:.2f} ms")
    print(f"  对应帧率:     {fps:.1f} FPS")
    print(f"{'=' * 60}")

    # 保存结果
    report = f"""推理时间测试报告
{'='*60}
GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}
CUDA: {torch.version.cuda if device.type == 'cuda' else 'N/A'}
模型: {weights_path}
输入尺寸: {IMAGE_SIZE} x {IMAGE_SIZE}
测试次数: {TEST_RUNS} (预热 {WARMUP_RUNS} 次)
{'='*60}
平均推理时间: {avg_time:.2f} ms
标准差:       {std_time:.2f} ms
中位数:       {median_time:.2f} ms
最小值:       {min_time:.2f} ms
最大值:       {max_time:.2f} ms
对应帧率:     {fps:.1f} FPS
{'='*60}
"""

    with open('results/inference_time_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告保存至: results/inference_time_report.txt")


if __name__ == '__main__':
    main()
