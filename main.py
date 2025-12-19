import sys, os
import json
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from utils.pose_utils import SELECTED_IDX

app = FastAPI()

# ==========================================
# 1. 模型載入
# ==========================================
model_path = "models/pose_sequence_classifier.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"✅ 模型已載入: {model_path}")
else:
    print(f"❌ 找不到模型檔案: {model_path}")
    model = None

# 載入標籤
labels_path = "models/labels.json"
idx_to_label = {}
if os.path.exists(labels_path):
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
            idx_to_label = {int(k): v for k, v in raw_labels.items()}
    except:
        pass

scaler_path = "models/scaler.npz"
mean, std = None, None
if os.path.exists(scaler_path):
    try:
        scaler = np.load(scaler_path)
        mean = scaler['mean']
        std = scaler['std']
        print(f"✅ Scaler 已載入: {scaler_path}")
    except Exception as e:
        print(f"❌ Scaler 載入失敗: {e}")
else:
    print(f"⚠️ 找不到 Scaler 檔案: {scaler_path} (預測可能會失準)")

if not idx_to_label:
    idx_to_label = {0: "hamstring_sweep", 1: "knee_hugs", 2: "neutral", 3: "walking_kicks"}

label_zh_map = {
    "hamstring_sweep": "執草",
    "knee_hugs": "抱腿",
    "walking_kicks": "踢腿",
    "neutral": "待機中",
    "buffering": "分析中...",
    "unknown": "未知動作"
}

# ==========================================
# 參數設定
# ==========================================
SEQ_LEN = 40
PREDICT_INTERVAL = 5

# ==========================================
# WebSocket 處理
# ==========================================
@app.websocket("/ws_predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"📱 手機端已連線: {websocket.client}")
    
    loop = asyncio.get_running_loop()
    
    # 建立一個容量只有 1 的隊列 (只存最新的一張圖)
    frame_queue = asyncio.Queue(maxsize=1)
    
    # 標記連線狀態，用來通知兩個 Task 停止
    connection_active = True

    # 初始化 MediaPipe (每個連線獨立)
    mp_pose = mp.solutions.pose
    local_pose_tracker = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, # 0=最快 (犧牲一點精度換速度)
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    local_seq_buffer = []
    frame_counter = 0

    # ----------------------------------------
    # 子任務 1: 專門負責「接收」 (Receiver)
    # ----------------------------------------
    async def receive_task():
        nonlocal connection_active
        try:
            while connection_active:
                # 接收 bytes
                data = await websocket.receive_bytes()
                
                # 如果隊列滿了 (代表處理端還在忙)，就把舊的 pop 掉
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                
                # 放新的進去
                await frame_queue.put(data)
                
        except WebSocketDisconnect:
            print("❌ 手機端斷開連線 (Receiver)")
            connection_active = False
        except Exception as e:
            print(f"Receiver 錯誤: {e}")
            connection_active = False

    # ----------------------------------------
    # 子任務 2: 專門負責「處理」 (Processor)
    # ----------------------------------------
    async def process_task():
        nonlocal connection_active, frame_counter, local_seq_buffer
        last_label_zh = "偵測中..."
        last_confidence = 0.0

        try:
            while connection_active:
                # 從隊列拿圖 (如果隊列是空的，這裡會等待，直到有圖進來)
                # 使用 wait_for 避免死鎖，如果 1 秒沒圖就檢查連線狀態
                try:
                    data = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if not connection_active: break
                    continue

                # --- 以下是原本的影像處理邏輯 ---
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None: continue
                

                height, width = img.shape[:2]
                target_width = 320
                scale = target_width / width
                target_height = int(height * scale)
                img = cv2.resize(img, (target_width, target_height))
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                
                # 在 Thread Pool 執行 MediaPipe (防止卡住主執行緒)
                results = await loop.run_in_executor(
                    None, 
                    lambda: local_pose_tracker.process(img_rgb)
                )
                
                final_landmarks = []         
                current_frame_keypoints = []

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    for landmark in landmarks:
                        final_landmarks.append({
                            "x": landmark.x, "y": landmark.y, "z": landmark.z, "v": landmark.visibility
                        })
                    current_frame_keypoints = np.array(
                        [[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in SELECTED_IDX]
                    ).flatten()
                else:
                    current_frame_keypoints = np.zeros(len(SELECTED_IDX) * 3)
                    
                if len(final_landmarks) > 0:
                    print(f"✅ 抓到了！ ({frame_counter})")
                else:
                    print(f"⚠️ 沒抓到人 - 回傳空數據 ({frame_counter})")

                local_seq_buffer.append(current_frame_keypoints)
                if len(local_seq_buffer) > SEQ_LEN:
                    local_seq_buffer.pop(0)

                frame_counter += 1
                should_predict = (len(local_seq_buffer) == SEQ_LEN) and (frame_counter % PREDICT_INTERVAL == 0)

                if should_predict and model is not None:
                    if np.sum(np.abs(local_seq_buffer[-1])) < 0.1:
                        last_label_zh = "請全身入鏡"
                        last_confidence = 0.0
                    else:
                        seq_array = np.array(local_seq_buffer, dtype=np.float32)
                        
                        # [新增] 套用標準化 (Normalization)
                        # 這一步至關重要！沒有它，模型就瞎了
                        if mean is not None and std is not None:
                            seq_array = (seq_array - mean) / std
                        
                        input_data = np.expand_dims(seq_array, axis=0)
                        
                        input_data = np.expand_dims(seq_array, axis=0)
                        
                        # 在 Thread Pool 執行模型預測
                        prediction = await loop.run_in_executor(
                            None, 
                            lambda: model.predict(input_data, verbose=0)
                        )
                        idx = int(np.argmax(prediction[0]))
                        label_en = idx_to_label.get(idx, "unknown")
                        last_label_zh = label_zh_map.get(label_en, label_en)
                        last_confidence = float(prediction[0][idx])
                        
                        if last_confidence < 0.6: 
                            last_label_zh = "動作不明確"

                # 回傳結果
                response_data = {
                    "label_zh": last_label_zh,   
                    "confidence": last_confidence,
                    "landmarks": final_landmarks 
                }
                
                await websocket.send_json(response_data)

        except Exception as e:
            print(f"Processor 錯誤: {e}")
            connection_active = False

    # ----------------------------------------
    # 主流程：並行執行接收與處理
    # ----------------------------------------
    # 使用 asyncio.gather 同時跑兩個無限迴圈
    # 只要其中一個斷線或出錯，就取消另一個
    try:
        await asyncio.gather(receive_task(), process_task())
    except Exception as e:
        print(f"WebSocket 主流程結束: {e}")
    finally:
        local_pose_tracker.close()
        print("🧹 資源釋放完畢")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)