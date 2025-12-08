import cv2
import mediapipe as mp
import numpy as np

# 可選 "right" 或 "left"
SIDE = "right"   # ← 想看左側就改成 "left"

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose

# 計算三點所形成的角度（在 b 點的夾角）
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
    print(f" 啟動{('右' if SIDE=='right' else '左')}側髖關節角度辨識模組...（按 q 鍵離開）")

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

                # 依側別選取節點（肩-髖-膝）→ 髖關節角度
                if SIDE == "right":
                    shoulder_lm = mp_pose.PoseLandmark.RIGHT_SHOULDER
                    hip_lm      = mp_pose.PoseLandmark.RIGHT_HIP
                    knee_lm     = mp_pose.PoseLandmark.RIGHT_KNEE
                else:
                    shoulder_lm = mp_pose.PoseLandmark.LEFT_SHOULDER
                    hip_lm      = mp_pose.PoseLandmark.LEFT_HIP
                    knee_lm     = mp_pose.PoseLandmark.LEFT_KNEE

                shoulder = [landmarks[shoulder_lm.value].x, landmarks[shoulder_lm.value].y]
                hip      = [landmarks[hip_lm.value].x,      landmarks[hip_lm.value].y]
                knee     = [landmarks[knee_lm.value].x,     landmarks[knee_lm.value].y]

                # 計算髖關節（屁股）角度：肩-髖-膝，角度在髖點
                angle = calculate_angle(shoulder, hip, knee)

                # 判斷角度是否小於 40
                if angle < 45:
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
                p2 = tuple(np.multiply(hip,      [520, 300]).astype(int))
                p3 = tuple(np.multiply(knee,     [520, 300]).astype(int))
                cv2.line(img, p1, p2, (255, 255, 0), 4)
                cv2.line(img, p2, p3, (255, 255, 0), 4)

                # 顯示角度與回饋文字
                cv2.putText(img, f'Hip angle: {int(angle)} deg', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(img, feedback, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # 顯示畫面
            win_name = f"{'Right' if SIDE=='right' else 'Left'} Hip Angle Check"
            cv2.imshow(win_name, img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# 執行程式
if __name__ == "__main__":
    run()
