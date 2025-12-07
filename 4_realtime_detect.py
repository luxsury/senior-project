import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from utils.pose_utils import extract_keypoints_from_frame
import time
import json
from PIL import ImageFont, ImageDraw, Image


# ============================================================
# 集中管理的可調參數】 —— 建議你所有調整都在這一段完成
# ============================================================

SEQ_LEN = 40                     # 模型看多少幀（越大越穩定，但反應越慢）
confidence_threshold = 0.70      # 信心度 ≥ 此值 → 可切換動作
stable_threshold = 0.90          # 信心度 ≥ 此值 → 視為穩定動作

camera_index = 0                 # 攝影機來源 (0外接/1內置)
model_complexity = 1             # Mediapipe Pose 模型複雜度：0=快、1=中等、2=準確
draw_pose = True                 # 是否在畫面中顯示骨架

font_size_main = 108             # 主文字（動作名稱）大小
font_size_category = 48          # 副標（分類）文字大小
font_size_warning = 80           # 警告字體大小
font_size_total = 68             # 下方總次數大小
font_size_small = 28             # 小字體大小


# ============================================================
# 中文文字繪製（Pillow）
# ============================================================
def draw_chinese_text(img, text, pos, color=(255,255,255), size=40, bold=False, outline=0):
    """
    OpenCV 原生不支援中文，使用 Pillow 繪製後再轉回 ndarray。
    支援：描邊、加粗、顏色。

    img : ndarray BGR frame
    text : 顯示文字
    pos : (x,y)
    size : 字體大小
    bold : 粗體效果
    outline : 外框厚度
    """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # 字型搜尋 (Windows / Mac / Linux)
    FONT_PATHS = [
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    ]
    font = None
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, size=size)
            break
    if font is None:
        font = ImageFont.load_default()

    x, y = pos

    # 文字描邊
    if outline > 0:
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                draw.text((x + dx, y + dy), text, font=font, fill=(0,0,0))

    # 粗體效果
    if bold:
        for dx, dy in [(1,0),(0,1),(-1,0),(0,-1)]:
            draw.text((x + dx, y + dy), text, font=font, fill=color)

    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)


# ============================================================
# Mediapipe Pose 初始化
# ============================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=model_complexity
)


# ============================================================
# 載入模型 + Normalization scaler + labels.json
# ============================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_sequence_classifier.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.npz")
LABEL_PATH = os.path.join(MODEL_DIR, "labels.json")

# 讀取模型
if not os.path.exists(MODEL_PATH):
    print(f"❌ 找不到模型檔案：{MODEL_PATH}")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ 模型已載入：{MODEL_PATH}")

# Normalization 參數
mean, std = None, None
if os.path.exists(SCALER_PATH):
    scaler = np.load(SCALER_PATH)
    mean = scaler['mean']
    std = scaler['std']
    print("✅ Scaler 已載入（mean/std）")
else:
    print("⚠️ 找不到 scaler.npz，模型效能可能降低")

# 讀取 labels.json
with open(LABEL_PATH, "r") as f:
    idx_to_label = json.load(f)

# 中文顯示對照
label_zh = {
    "hamstring_sweep": "執草",
    "knee_hugs": "抱腿",
    "walking_kicks": "踢腿",
    "neutral": "無動作/待機中"
}

label_category = {
    "hamstring_sweep": "暖身 / 腿後側",
    "knee_hugs": "暖身 / 臀腿",
    "walking_kicks": "暖身 / 動態腿部",
    "neutral": "無動作 / 待機"
}


# ============================================================
# 讀取攝影機
# ============================================================
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

print("🎥 開始即時偵測（按 Q 結束）...")

# ============================================================
# 狀態變數初始化（動作狀態機）
# ============================================================

seq_buffer = []                      # 存放過去 SEQ_LEN 幀的關鍵點
fps = 0.0
prev_time = time.time()

total_count = 0                      # 所有動作累計次數
per_label_count = {lbl: 0 for lbl in idx_to_label.values()}  # 每個動作計數

in_action = False                    # 是否目前「正在動作中」
current_action = None               # 當前動作名稱


# 顏色依信心度選擇
def color_by_conf(conf):
    if conf >= stable_threshold:
        return (102, 255, 102)  # 綠
    if conf >= confidence_threshold:
        return (102, 255, 255)  # 黃
    return (80, 80, 255)        # 紅


# ============================================================
# 主推論迴圈（Realtime）
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 鏡像
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    pose_detected = results.pose_landmarks is not None

    # 預設 UI 顯示內容
    action_label_zh = "偵測中..."
    action_category = "--"
    confidence = 0.0
    alert = False
    show_warning = False
    text_color = (102, 255, 102)

    # ===== 繪製骨架（可關閉）=====
    if draw_pose and pose_detected:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(180,180,180), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(51,0,153), thickness=20, circle_radius=2)
        )

    # ===== 取得當幀的關鍵點 =====
    keypoints = extract_keypoints_from_frame(frame)
    seq_buffer.append(keypoints)

    if len(seq_buffer) > SEQ_LEN:
        seq_buffer.pop(0)

    # ===== 當蒐集到足夠幀數時，執行模型推論 =====
    if len(seq_buffer) == SEQ_LEN and pose_detected:

        seq_array = np.array(seq_buffer, dtype=np.float32)

        # Normalization
        if mean is not None and std is not None:
            seq_array = (seq_array - mean) / std

        # 增加 batch 維度
        seq_input = seq_array.reshape(1, SEQ_LEN, -1)

        # 模型推論
        prediction = model.predict(seq_input, verbose=0)
        idx = int(np.argmax(prediction))
        label = idx_to_label[str(idx)]
        confidence = float(np.max(prediction))

        # UI 顯示內容
        action_label_zh = label_zh[label]
        action_category = label_category[label]
        text_color = color_by_conf(confidence)

        if confidence < confidence_threshold:
            alert = True
            show_warning = True

        # ============================================================
        #【動作狀態機】—— 用來判斷「開始動作」與「動作完成」
        # ============================================================

        # 1. 當信心度 ≥ 門檻 → 更新當前動作
        if confidence >= confidence_threshold:

            if current_action != label:
                # 偵測到新動作
                current_action = label
                in_action = True
            else:
                # 持續同一動作
                in_action = True

        # 2. 信心度 < 門檻 → 視為「動作結束」，開始計數
        else:
            if in_action and current_action is not None:
                per_label_count[current_action] += 1
                total_count += 1

                # 重置狀態
                in_action = False
                current_action = None

    # 若根本沒看到人體
    elif not pose_detected:
        alert = True
        show_warning = True
        in_action = False
        current_action = None

    # ============================================================
    # UI 繪製（動作名稱、分類、計數、警告、FPS）
    # ============================================================

    # 畫紅框提示用戶調整位置
    if alert:
        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 18)

    # 動作名稱
    frame = draw_chinese_text(frame, 
                              f"{action_label_zh} ({confidence:.2f})",
                              (30, 40),
                              color=text_color, size=font_size_main, bold=True, outline=3)

    # 動作分類
    frame = draw_chinese_text(frame, 
                              f"分類：{action_category}",
                              (30, 200),
                              color=(255,255,255), size=font_size_category, bold=True, outline=3)

    # 計數
    h = frame.shape[0]
    frame = draw_chinese_text(frame, f"總次數：{total_count}",
                              (30, h - 130),
                              color=(102,255,255), size=font_size_total, bold=True, outline=3)

    # 每個動作計數
    per_txt = (
        f"執草：{per_label_count.get('hamstring_sweep',0)}  "
        f"抱腿：{per_label_count.get('knee_hugs',0)}  "
        f"踢腿：{per_label_count.get('walking_kicks',0)}"
    )
    frame = draw_chinese_text(frame, per_txt, (30, h - 50),
                              color=(200,200,200), size=font_size_small)

    # 警告文字
    if show_warning:
        frame = draw_chinese_text(frame, "動作不明 / 請調整位置",
                                  (30, 300),
                                  color=(102,102,255), size=font_size_warning,
                                  bold=True, outline=2)

    # FPS 計算
    now = time.time()
    dt = now - prev_time
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1 / dt)
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 170, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180,255,180), 2)

    # 顯示畫面
    cv2.imshow("Pose Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("👋 結束即時偵測。")
