import mediapipe as mp
import numpy as np
import cv2

# 1. 初始化 MediaPipe (全域變數)
mp_pose = mp.solutions.pose

# ✅ 修正變數名稱為 pose_tracker，避免 NameError
pose_tracker = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,       # 1=預設 (平衡), 0=快, 2=準
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 你的自定義關鍵點（16點：四肢＋軀幹）
# 為了讓 main.py 也可以引用這個常數，請保留在這裡
SELECTED_IDX = [
    11, 12,  # 肩膀
    13, 14,  # 手肘
    15, 16,  # 手腕
    23, 24,  # 臀部
    25, 26,  # 膝蓋
    27, 28,  # 腳踝
    29, 30,  # 腳跟
    31, 32   # 腳尖
]

def extract_keypoints_from_frame(frame):
    """
    從單一影格提取關鍵點 (主要給 collect_data.py 使用)
    注意：main.py 為了效能已經內建這段邏輯，不會呼叫此函式。
    """
    # 轉 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False 
    
    # 2. 進行偵測 
    # ✅ 這裡使用上面定義的 pose_tracker
    results = pose_tracker.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 抓取指定 16 點
        selected_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in SELECTED_IDX])
        return selected_points.flatten()     
    else:
        # 沒偵測到人時回傳 0
        return np.zeros(len(SELECTED_IDX) * 3)