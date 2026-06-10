import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.datasets.basketball_dataset import BasketballDataset

# ==================== 配置 ====================
train_list = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\train.txt'  # 已合并
val_list = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\test.txt'  # 验证集（124张）

image_size = 640
batch_size = 2
num_workers = 4
pin_memory = False
num_epochs = 50
lr = 1e-4
weight_decay = 0.0005
# =================================================

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def check_target_valid(target):
    boxes = target['boxes']
    if len(boxes) == 0:
        return True
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if (widths <= 0).any() or (heights <= 0).any():
        return False
    if (boxes < 0).any():
        return False
    return True

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, is_best=False, filename='weights/expC_with_Tracer_latest.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, filename)
    if is_best:
        torch.save(model.state_dict(), 'weights/expC_with_Tracer_best.pth')
        print(f"✅ 最佳模型已保存（验证损失 {best_val_loss:.4f}）")
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer, scheduler, device):
    if not os.path.exists(filename):
        print("未找到 checkpoint，从头开始训练")
        return 0, float('inf')
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"恢复训练，从 epoch {start_epoch} 开始，当前最佳验证损失 {best_val_loss:.4f}")
    return start_epoch, best_val_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"初始显存占用: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(">>> 加载训练集...")
    train_dataset = BasketballDataset(train_list, image_size=image_size)
    print(f"训练集样本数: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print("错误: 训练集为空，请检查数据路径。")
        return

    print(">>> 加载验证集...")
    val_dataset = BasketballDataset(val_list, image_size=image_size)
    print(f"验证集样本数: {len(val_dataset)}")
    if len(val_dataset) == 0:
        print("错误: 验证集为空，请检查数据路径。")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(">>> 加载预训练模型...")
    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

    num_classes = 2
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.cls_logits.in_channels
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, padding=1
    )
    model.head.classification_head.num_classes = num_classes
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    os.makedirs('weights', exist_ok=True)
    checkpoint_path = 'weights/expC_with_Tracer_latest.pth'
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

    print(">>> 开始训练（实验c：大数据+Tracer，优化策略）...")
    patience = 10
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, targets) in enumerate(pbar):
            valid = True
            for t in targets:
                if not check_target_valid(t):
                    valid = False
                    break
            if not valid:
                print(f"\n⚠️ 跳过 batch {batch_idx}：目标框无效")
                continue

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                print(f"\n⚠️ 跳过 batch {batch_idx}，损失异常")
                optimizer.zero_grad()
                continue

            scaler.scale(losses).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_loss = losses.item()
            train_loss += batch_loss
            pbar.set_postfix({'loss': f"{batch_loss:.2f}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.4f}")

        # 每5个epoch验证一次
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="验证"):
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    output = model(images, targets)
                    if isinstance(output, dict):
                        losses = sum(output.values())
                    else:
                        losses = torch.tensor(0.0, device=device)
                    val_loss += losses.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} 平均验证损失: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, is_best=True, filename=checkpoint_path)
            else:
                patience_counter += 1
                save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, is_best=False, filename=checkpoint_path)
                if patience_counter >= patience:
                    print(f"早停于 epoch {epoch+1}")
                    break
        else:
            # 非验证轮次也保存最新 checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, is_best=False, filename=checkpoint_path)

        scheduler.step()

        # 每10个epoch保存独立模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'weights/expC_with_Tracer_epoch{epoch+1}.pth')
            print(f"独立模型已保存: weights/expC_with_Tracer_epoch{epoch+1}.pth")

    print("训练完成！最终模型已保存至 weights/expC_with_Tracer_final.pth")
    torch.save(model.state_dict(), 'weights/expC_with_Tracer_final.pth')

if __name__ == '__main__':
    train()