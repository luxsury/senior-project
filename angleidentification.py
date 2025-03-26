import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# 計算角度的函數
def calculate_angle(a, b, c):
    # 使用向量運算計算角度
    a = np.array(a)  # 向量A
    b = np.array(b)  # 向量B
    c = np.array(c)  # 向量C

    ab = a - b  # 向量AB
    bc = c - b  # 向量BC

    # 計算兩個向量的點積
    dot_product = np.dot(ab, bc)
    # 計算兩個向量的模長
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    
    # 計算角度 (弧度制)
    angle = np.arccos(dot_product / (norm_ab * norm_bc))
    angle = np.degrees(angle)  # 轉換為角度
    return angle

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        enable_segmentation=True,
        min_tracking_confidence=0.5) as pose:
        
        if not cap.isOpened():
            print("無法打開攝像頭")
            exit()

        while True:
            ret, img = cap.read()
            if not ret:
                print("無法接收視頻幀") 
                break  
            
            img = cv2.resize(img, (520, 300))
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = pose.process(img2)

            try:
                condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.5
                bg = np.zeros_like(img)
                img = np.where(condition, img, bg)
            except:
                pass

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # 獲取肩膀、肘部、手腕的座標
                landmarks = result.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # 計算角度
                angle = calculate_angle(shoulder, elbow, wrist)

                # 顯示角度
                cv2.putText(img, f'Angle: {int(angle)}', 
                            tuple(np.multiply(elbow, [520, 300]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # 繪製所有landmarks但不顯示座標
                for landmark in result.pose_landmarks.landmark:  
                    h, w, c = img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)

                    # 只繪製標誌點，並且不顯示座標
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

                    # 打印每個landmark的座標到終端
                    print(f"Landmark: x = {landmark.x:.4f}, y = {landmark.y:.4f}")

            cv2.imshow('Pose Estimation', img)

            if cv2.waitKey(5) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
