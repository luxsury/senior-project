#先開啟虛擬機 conda activate pose-env
#關閉虛擬環境 conda deactivate
#確認python路徑 which python
#切換python位置 export PATH="/opt/anaconda3/envs/pose-env/bin:$PATH"
#測試環境 python -c "import tensorflow as tf; import mediapipe as mp; import cv2; print(tf.__version__, mp.__version__, cv2.__version__)"
#圖形呈現測試 python scripts/eval_model.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pose_utils import extract_keypoints_from_frame
import cv2
import numpy as np
import mediapipe as mp

# === 正確設定專案根目錄 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data/raw_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed_keypoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keypoints_all = []

    print(f"📹 開始處理影片：{os.path.basename(video_path)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 1 == 0:
            keypoints = extract_keypoints_from_frame(frame)
            keypoints_all.append(keypoints)
            if frame_count % 50 == 0:
                print(f"  🔹 已處理 {frame_count} 幀", end="\r")

        frame_count += 1

    cap.release()

    # ⭐ 建立子資料夾
    save_path = os.path.join(OUTPUT_DIR, f"{label}.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.save(save_path, np.array(keypoints_all))
    print(f"\n✅ 完成影片：{os.path.basename(video_path)}，共 {len(keypoints_all)} 幀。")


if __name__ == "__main__":
    videos = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(DATA_DIR)
        for f in filenames if f.lower().endswith((".mp4", ".mov", ".m4v"))
    ]

    print(f"🎬 偵測到 {len(videos)} 支影片")
    print(f"前 5 支影片示範：{videos[:5]}")

    for file in videos:
        label = os.path.splitext(os.path.relpath(file, DATA_DIR))[0]
        process_video(file, label)

    print("📦 全部影片處理完成！")