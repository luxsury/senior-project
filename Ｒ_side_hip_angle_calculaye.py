import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose

# 計算三點所形成的角度（肩-髖-膝）
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

def run():
    print(" 啟動右腰角度辨識模組...（按 q 鍵離開）")

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        if not cap.isOpened():
            print(" 無法打開攝像頭")
            return

        while True:
            ret, img = cap.read()
            if not ret:
                print(" 無法接收視頻幀")
                break

            img = cv2.resize(img, (520, 300))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                # 抓取右肩、右髖、右膝座標
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # 計算角度
                angle = calculate_angle(shoulder, hip, knee)

                # 判斷角度是否小於 40
                if angle < 40:
                    print("right")
                    feedback = "right"
                    color = (0, 255, 0)
                else:
                    print("wrong")
                    feedback = "wrong"
                    color = (0, 0, 255)

                # 繪製關鍵點
                for pt in [shoulder, hip, knee]:
                    x, y = int(pt[0] * 520), int(pt[1] * 300)
                    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

                # 畫線連接
                p1 = tuple(np.multiply(shoulder, [520, 300]).astype(int))
                p2 = tuple(np.multiply(hip, [520, 300]).astype(int))
                p3 = tuple(np.multiply(knee, [520, 300]).astype(int))
                cv2.line(img, p1, p2, (255, 255, 0), 4)
                cv2.line(img, p2, p3, (255, 255, 0), 4)

                # 顯示角度與回饋文字
                cv2.putText(img, f'Waist angle: {int(angle)} deg', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(img, feedback, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # 顯示畫面
            cv2.imshow('Right Waist Angle Check', img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# 執行程式
if __name__ == "__main__":
    run()