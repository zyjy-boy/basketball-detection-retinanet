"""
将 Label Studio 导出的 JSON 转换为 GT 标注格式
输入：Label Studio 导出的 JSON（Export → JSON）
输出：results/gt_annotations/<视频名>.json

输出格式（与 compare_trajectory_batch.py 兼容）：
{"frame_0000": [{"x1": 100, "y1": 200, "x2": 150, "y2": 250}], "frame_0005": [], ...}

用法：
  python convert_labelstudio_to_gt.py
"""

import json
import os
import re
import glob

# ==================== 配置 ====================
# Label Studio 导出的 JSON 文件路径
EXPORT_DIR = r'results/label_studio_export'

LABEL_STUDIO_EXPORTS = {
    'test_video':   os.path.join(EXPORT_DIR, 'text.json'),
    'test_video_03': os.path.join(EXPORT_DIR, 'text_03.json'),
    'test_video_04': os.path.join(EXPORT_DIR, 'text_04.json'),
}

OUTPUT_DIR = r'results/gt_annotations'
# ================================================


def extract_frame_name(task):
    """从 task 中提取帧名，兼容两种格式：
    格式1（test_video）: file_upload = '6a6ae8c9-frame_0460.jpg'
    格式2（test_video_03/04）: data.image = 'http://localhost:8765/test_video_03/frame_0000.jpg'
    """
    # 方式1: file_upload 字段
    file_upload = task.get('file_upload', '')
    if file_upload:
        match = re.search(r'frame_\d+', file_upload)
        if match:
            return match.group()

    # 方式2: data.image URL
    data = task.get('data', {})
    image_url = data.get('image', '')
    if image_url:
        match = re.search(r'frame_\d+', image_url)
        if match:
            return match.group()

    return None


def convert_labelstudio_to_gt(json_path, video_name):
    """将 Label Studio 导出 JSON 转为 GT 格式"""
    if not os.path.exists(json_path):
        print(f"  ✗ 未找到: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gt_data = {}
    annotated_count = 0
    empty_count = 0
    skipped_count = 0

    for task in data:
        # 提取帧名
        frame_name = extract_frame_name(task)
        if frame_name is None:
            skipped_count += 1
            continue

        # 提取标注结果
        annotations = task.get('annotations', [])
        boxes = []

        for ann in annotations:
            # 跳过被取消的标注（没有篮球的帧）
            if ann.get('was_cancelled', False):
                continue

            results = ann.get('result', [])
            for r in results:
                if r.get('type') != 'rectanglelabels':
                    continue

                value = r['value']
                original_width = r['original_width']
                original_height = r['original_height']

                # Label Studio 坐标是百分比，转为像素
                x_pct = value['x']
                y_pct = value['y']
                w_pct = value['width']
                h_pct = value['height']

                x1 = x_pct / 100 * original_width
                y1 = y_pct / 100 * original_height
                x2 = x1 + w_pct / 100 * original_width
                y2 = y1 + h_pct / 100 * original_height

                boxes.append({
                    'x1': round(x1, 2),
                    'y1': round(y1, 2),
                    'x2': round(x2, 2),
                    'y2': round(y2, 2)
                })

        gt_data[frame_name] = boxes
        if len(boxes) > 0:
            annotated_count += 1
        else:
            empty_count += 1

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'{video_name}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {video_name}: {len(gt_data)}帧, 有篮球{annotated_count}帧, 无篮球{empty_count}帧, 跳过{skipped_count}帧")
    print(f"    保存至: {output_path}")


def main():
    print("=" * 60)
    print("Label Studio 导出 → GT 标注格式转换")
    print("=" * 60)

    for video_name, json_path in LABEL_STUDIO_EXPORTS.items():
        print(f"\n处理: {video_name}")
        convert_labelstudio_to_gt(json_path, video_name)

    print(f"\n{'=' * 60}")
    print("完成！")
    print("现在可以运行 compare_trajectory_batch.py 进行完整对比分析")


if __name__ == '__main__':
    main()
