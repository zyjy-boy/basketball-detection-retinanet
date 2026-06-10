"""
检查 deepsportradar 数据集 train.txt 与 test.txt 是否存在图片重叠
"""

import os


# =================== 配置 ===================
TRAIN_LIST = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\train.txt'
TEST_LIST = r'.\data\SportsMOT\deepsportradar-DatasetNinja\labels\test.txt'
# ==========================================


def extract_keys(list_path):
    """从 list 文件中提取图片的唯一标识（文件名，不含路径和扩展名）"""
    keys = set()
    if not os.path.exists(list_path):
        print(f"⚠ 文件不存在: {list_path}")
        return keys
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # list_file 格式：每行 "图片路径 标签路径"
            img_path = line.split(maxsplit=1)[0]
            # 取文件名（不含扩展名）作为唯一标识
            name = os.path.splitext(os.path.basename(img_path))[0]
            keys.add(name)
    return keys


def main():
    print("=" * 60)
    print("训练集 / 验证集重叠检查")
    print("=" * 60)

    train_keys = extract_keys(TRAIN_LIST)
    test_keys = extract_keys(TEST_LIST)

    print(f"\n训练集图片数: {len(train_keys)}")
    print(f"验证集图片数: {len(test_keys)}")

    overlap = train_keys & test_keys
    overlap_rate_train = len(overlap) / len(train_keys) * 100 if train_keys else 0
    overlap_rate_test = len(overlap) / len(test_keys) * 100 if test_keys else 0

    if len(overlap) == 0:
        print(f"\n✅ 训练集与验证集无重叠")
    else:
        print(f"\n❌ 发现 {len(overlap)} 张重叠图片")
        print(f"   占训练集: {overlap_rate_train:.1f}%")
        print(f"   占验证集: {overlap_rate_test:.1f}%")
        print(f"\n重叠文件名（前 50 个）:")
        for i, name in enumerate(sorted(overlap)[:50]):
            print(f"  {i+1}. {name}")
        if len(overlap) > 50:
            print(f"  ... 还有 {len(overlap) - 50} 个")

    print(f"\n{'=' * 60}")
    print(f"训练集独有: {len(train_keys - test_keys)} 张")
    print(f"验证集独有: {len(test_keys - train_keys)} 张")
    print(f"重叠:       {len(overlap)} 张")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
