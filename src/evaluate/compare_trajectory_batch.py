"""
脚本3：批量检测对比分析
功能：将视频SORT检测、抽帧静态检测与真实标注进行对比
指标：漏检率（Miss Rate）、误检率（False Positive Rate）
输入：
  - results/video_detections/<视频名>.json  （视频SORT检测）
  - results/static_detections/<视频名>.json （静态检测）
  - results/gt_annotations/<视频名>.json    （真实标注）
输出：
  - results/trajectory_comparison/<视频名>_metrics.txt
"""

import json
import os
import numpy as np
import cv2

# ==================== 配置 ====================
FRAMES_DIR = r'.\data\frames'
OUTPUT_DIR = r'results/trajectory_comparison'

VIDEO_NAMES = ['test_video', 'test_video_03', 'test_video_04']
SAMPLE_INTERVAL = 5

VIDEO_DET_FILES = {
    'test_video':   r'results/video_detections.json',
    'test_video_03': r'results/video_detections_03.json',
    'test_video_04': r'results/video_detections_04.json',
}
# ================================================


def load_json(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_frame_names(video_name):
    frame_dir = os.path.join(FRAMES_DIR, video_name)
    if not os.path.exists(frame_dir):
        return []
    return [f.replace('.jpg', '') for f in sorted(os.listdir(frame_dir)) if f.endswith('.jpg')]


def extract_center_from_video_det(video_det, frame_name):
    if video_det is None or frame_name not in video_det:
        return None
    dets = video_det[frame_name]
    if not dets or len(dets) == 0:
        return None
    d = dets[0]
    return ((d['x1'] + d['x2']) / 2, (d['y1'] + d['y2']) / 2)


def extract_center_from_static_det(static_det, frame_name):
    if frame_name not in static_det:
        return None
    d = static_det[frame_name]
    if d is None:
        return None
    return (d['cx'], d['cy'])


def extract_center_from_gt(gt_data, frame_name):
    if gt_data is None or frame_name not in gt_data:
        return None
    d = gt_data[frame_name]
    if not d or len(d) == 0:
        return None
    box = d[0]
    return ((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2)


def compute_miss_fp_rate(pred_points, gt_points, threshold=20):
    """
    计算漏检率和误检率
    - 漏检率 (Miss Rate): GT有篮球但检测没有的比例
    - 误检率 (FP Rate): GT没有篮球但检测有的比例
    """
    gt_total = sum(1 for g in gt_points if g is not None)  # GT中有篮球的帧数
    gt_empty = sum(1 for g in gt_points if g is None)     # GT中没有篮球的帧数

    miss = 0      # GT有但检测没有
    fp = 0        # GT没有但检测有

    for g, p in zip(gt_points, pred_points):
        if g is not None and p is None:
            miss += 1
        elif g is None and p is not None:
            fp += 1

    miss_rate = miss / gt_total if gt_total > 0 else 0
    fp_rate = fp / gt_empty if gt_empty > 0 else 0

    return {
        'gt_total': gt_total,      # GT有篮球帧数
        'gt_empty': gt_empty,       # GT无篮球帧数
        'miss': miss,               # 漏检帧数
        'fp': fp,                   # 误检帧数
        'miss_rate': miss_rate,     # 漏检率
        'fp_rate': fp_rate,         # 误检率
    }


def process_video(video_name):
    print(f"\n{'=' * 50}")
    print(f"处理: {video_name}")
    print(f"{'=' * 50}")

    video_det_path = VIDEO_DET_FILES.get(video_name)
    video_det = load_json(video_det_path) if video_det_path else None
    static_det = load_json(os.path.join(r'results/static_detections', f'{video_name}.json'))
    gt_data = load_json(os.path.join(r'results/gt_annotations', f'{video_name}.json'))

    frame_names = get_frame_names(video_name)
    if not frame_names:
        print(f"  ✗ 未找到帧图片")
        return

    print(f"  帧数: {len(frame_names)}")
    print(f"  视频检测: {'✓' if video_det else '✗ 未找到'}")
    print(f"  静态检测: {'✓' if static_det else '✗ 未找到'}")
    print(f"  真实标注: {'✓' if gt_data else '✗ 未标注'}")

    video_points = []
    static_points = []
    gt_points = []

    for fn in frame_names:
        video_points.append(extract_center_from_video_det(video_det, fn))
        static_points.append(extract_center_from_static_det(static_det, fn))
        gt_points.append(extract_center_from_gt(gt_data, fn))

    # 检测率统计
    video_valid = sum(1 for p in video_points if p is not None)
    static_valid = sum(1 for p in static_points if p is not None)
    gt_valid = sum(1 for p in gt_points if p is not None)

    # 位置匹配（两者都检测到时）
    match_count = 0
    both_valid = 0
    for v, s in zip(video_points, static_points):
        if v is not None and s is not None:
            both_valid += 1
            dist = np.sqrt((v[0] - s[0]) ** 2 + (v[1] - s[1]) ** 2)
            if dist < 20:
                match_count += 1

    # 漏检率和误检率
    video_metrics = None
    static_metrics = None
    if gt_data is not None and gt_valid > 0:
        video_metrics = compute_miss_fp_rate(video_points, gt_points)
        static_metrics = compute_miss_fp_rate(static_points, gt_points)

    # ====== 输出报告 ======
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, f'{video_name}_metrics.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"检测对比分析: {video_name}\n")
        f.write(f"帧数: {len(frame_names)} (每{SAMPLE_INTERVAL}帧采样)\n")
        f.write("=" * 60 + "\n\n")

        f.write("【检测统计】\n")
        f.write(f"  视频SORT检测: {video_valid}/{len(frame_names)} ({video_valid/len(frame_names)*100:.1f}%)\n")
        f.write(f"  抽帧静态检测: {static_valid}/{len(frame_names)} ({static_valid/len(frame_names)*100:.1f}%)\n")
        if gt_data is not None:
            f.write(f"  真实标注:     {gt_valid}/{len(frame_names)} ({gt_valid/len(frame_names)*100:.1f}%)\n")
        else:
            f.write(f"  真实标注:     未标注（待补充）\n")

        f.write(f"\n【视频检测 vs 静态检测】\n")
        f.write(f"  两者都检测到: {both_valid} 帧\n")
        f.write(f"  位置匹配(<20px): {match_count}/{both_valid} ({match_count/max(both_valid,1)*100:.1f}%)\n")

        if video_metrics and static_metrics:
            f.write(f"\n【与真实标注对比】\n")
            f.write(f"  真实有篮球帧数: {video_metrics['gt_total']}\n")
            f.write(f"  真实无篮球帧数: {video_metrics['gt_empty']}\n")
            f.write(f"\n")
            f.write(f"  {'指标':<20}| {'视频SORT':<16}| {'抽帧静态':<16}\n")
            f.write(f"  {'-'*52}\n")
            f.write(f"  {'漏检帧数':<20}| {video_metrics['miss']:<16}| {static_metrics['miss']:<16}\n")
            f.write(f"  {'误检帧数':<20}| {video_metrics['fp']:<16}| {static_metrics['fp']:<16}\n")
            f.write(f"  {'漏检率':<20}| {video_metrics['miss_rate']:<16.4f}| {static_metrics['miss_rate']:<16.4f}\n")
            f.write(f"  {'误检率':<20}| {video_metrics['fp_rate']:<16.4f}| {static_metrics['fp_rate']:<16.4f}\n")
        else:
            f.write(f"\n【与真实标注对比】\n")
            f.write(f"  待补充真实标注后计算\n")

        # 逐帧数据
        f.write(f"\n【逐帧数据】\n")
        f.write(f"  {'帧名':<16}| {'视频检测':<12}| {'静态检测':<12}| {'真实标注':<12}\n")
        f.write(f"  {'-'*52}\n")
        for i, fn in enumerate(frame_names):
            v_str = "✓" if video_points[i] else "✗"
            s_str = "✓" if static_points[i] else "✗"
            if gt_data is not None:
                g_str = "✓" if gt_points[i] else "✗"
            else:
                g_str = "-"
            f.write(f"  {fn:<16}| {v_str:<12}| {s_str:<12}| {g_str:<12}\n")

    print(f"\n  报告保存至: {report_path}")

    # 控制台输出
    print(f"\n  检测统计:")
    print(f"    视频SORT: {video_valid}/{len(frame_names)} ({video_valid/len(frame_names)*100:.1f}%)")
    print(f"    抽帧静态: {static_valid}/{len(frame_names)} ({static_valid/len(frame_names)*100:.1f}%)")
    print(f"    两者匹配: {match_count}/{both_valid}")
    if video_metrics and static_metrics:
        print(f"\n  与GT对比:")
        print(f"    视频SORT  漏检率={video_metrics['miss_rate']:.4f} ({video_metrics['miss']}/{video_metrics['gt_total']}), 误检率={video_metrics['fp_rate']:.4f} ({video_metrics['fp']}/{video_metrics['gt_empty']})")
        print(f"    抽帧静态  漏检率={static_metrics['miss_rate']:.4f} ({static_metrics['miss']}/{static_metrics['gt_total']}), 误检率={static_metrics['fp_rate']:.4f} ({static_metrics['fp']}/{static_metrics['gt_empty']})")


def main():
    print("=" * 60)
    print("批量检测对比分析（漏检率 + 误检率）")
    print(f"视频: {VIDEO_NAMES}")
    print(f"采样间隔: 每{SAMPLE_INTERVAL}帧")
    print("=" * 60)

    for video_name in VIDEO_NAMES:
        process_video(video_name)

    print(f"\n{'=' * 60}")
    print("全部完成！")


if __name__ == '__main__':
    main()
