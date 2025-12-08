import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# 計算三點形成的夾角
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    bc = c - b
    dot_product = np.dot(ab, bc)
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    if norm_ab == 0 or norm_bc == 0:
        return 0
    angle = np.arccos(np.clip(dot_product / (norm_ab * norm_bc), -1.0, 1.0))
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    enable_segmentation=False,
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # 取得右手肩膀、手肘、手腕座標
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # 計算角度
            angle = calculate_angle(shoulder, elbow, wrist)

            # ✅ 判斷角度是否小於 40
            if angle < 40:
                print("right")
                feedback = "right"
                color = (0, 255, 0)
            else:
                print("wrong")
                feedback = "wrong"
                color = (0, 0, 255)

            # 繪製三個點
            for pt in [shoulder, elbow, wrist]:
                x, y = int(pt[0] * 520), int(pt[1] * 300)
                cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(img, f"({pt[0]:.2f}, {pt[1]:.2f})",
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # 繪製兩條連線（肩→肘、肘→腕）
            p1 = tuple(np.multiply(shoulder, [520, 300]).astype(int))
            p2 = tuple(np.multiply(elbow, [520, 300]).astype(int))
            p3 = tuple(np.multiply(wrist, [520, 300]).astype(int))
            cv2.line(img, p1, p2, (255, 255, 0), 2)
            cv2.line(img, p2, p3, (255, 255, 0), 2)

            # 顯示角度與回饋文字
            cv2.putText(img, f'Angle: {int(angle)}°', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, feedback, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        cv2.imshow('Right Arm Angle Check', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
