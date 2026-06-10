"""
脚本1：批量视频抽帧
功能：对指定视频逐帧提取图片，每隔5帧保存一帧
输入：data/videos/ 下的视频文件
输出：data/frames/<视频名>/frame_0000.jpg, frame_0005.jpg, ...
"""

import cv2
import os
import glob

# ==================== 配置 ====================
VIDEO_DIR = r'.\data\videos'
FRAMES_DIR = r'.\data\frames'
SAMPLE_INTERVAL = 5  # 每隔5帧抽取一帧
VIDEO_NAMES = ['test_video', 'test_video_03', 'test_video_04']
# ================================================


def extract_frames(video_path, output_dir, sample_interval):
    """从视频中按间隔抽取帧"""
    # 清空输出目录，避免残留文件
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith('.jpg'):
                os.remove(os.path.join(output_dir, f))
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ 无法打开: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            filename = f"frame_{frame_idx:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"  ✓ {os.path.basename(video_path)}: {total_frames}帧, 抽取{saved_count}帧, {width}x{height}, {fps:.1f}FPS")
    return saved_count


def main():
    print("=" * 60)
    print("批量视频抽帧（每隔{}帧）".format(SAMPLE_INTERVAL))
    print(f"视频目录: {VIDEO_DIR}")
    print(f"输出目录: {FRAMES_DIR}")
    print(f"视频列表: {VIDEO_NAMES}")
    print("=" * 60)

    total_saved = 0
    for video_name in VIDEO_NAMES:
        # 查找视频文件（支持 mp4/avi/mov）
        pattern = os.path.join(VIDEO_DIR, video_name + '.*')
        video_files = glob.glob(pattern)

        if not video_files:
            print(f"\n  ✗ 未找到视频: {video_name}")
            continue

        video_path = video_files[0]
        output_dir = os.path.join(FRAMES_DIR, video_name)
        print(f"\n处理: {os.path.basename(video_path)}")
        count = extract_frames(video_path, output_dir, SAMPLE_INTERVAL)
        total_saved += count

    print(f"\n{'=' * 60}")
    print(f"完成！共抽取 {total_saved} 帧")
    print(f"输出目录: {FRAMES_DIR}")
    for video_name in VIDEO_NAMES:
        frame_dir = os.path.join(FRAMES_DIR, video_name)
        if os.path.exists(frame_dir):
            n = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
            print(f"  {video_name}: {n} 帧")


if __name__ == '__main__':
    main()
