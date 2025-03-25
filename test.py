import cv2
import mediapipe as mp
mp_drawing = mp.solution.drawing_utils
mp_drawing_styles = mp.solution.drawing_styles
mp_pose = mp.solution.pose

cap = cv2.VideoCapture(0)

with mp_pose.pose(
    min_detection_confidence=0.5,
    enable_segmentation=True,
    min_tracking_confidence=0.5) as pose:
    if not cap.isOpened():
        print("cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
    if not ret:
        print("cannot receive frame")
    break
    img = cv2.resize(img,(520,300))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.pose.process(img2)
    try:
        condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 1
        img = np.where(condition, img, bg)
    except:
        pass
    mp_drawing.draw_landmarks(
        img,
        result.pose_landmarks,
        mp_pose.POSE_CONNECTIONs,
        landmark_drawing_spec=mp_drawing_style.get_default_pose_landmarks_style())

    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(5) == ord('q'):
    break
    cap.release()
    cv2.destoryAllWindows()