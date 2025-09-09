import os
import mediapipe as mp
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0=lite, 1=full, 2=heavy
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def judgeAngleSide(ear, shoulder, hip, knee, image):
    earArea = np.array([ear.x, ear.y])
    shoulderArea = np.array([shoulder.x, shoulder.y])
    hipArea = np.array([hip.x, hip.y])
    kneeArea = np.array([knee.x, knee.y])
    
    # vectol
    vecESH_a = earArea - shoulderArea
    vecESH_c = hipArea - shoulderArea
    vecSHK_a = shoulderArea - hipArea
    vecSHK_c = kneeArea - hipArea

    # cosine
    cos_ESH = np.inner(vecESH_a, vecESH_c) / (np.linalg.norm(vecESH_a) * np.linalg.norm(vecESH_c))
    cos_SHK = np.inner(vecSHK_a, vecSHK_c) / (np.linalg.norm(vecSHK_a) * np.linalg.norm(vecSHK_c))

    # angle
    degree1 = np.rad2deg(np.arccos(cos_ESH))
    degree2 = np.rad2deg(np.arccos(cos_SHK))

    judge_ESH = abs(180 - degree1)
    judge_SHK = abs(180 - degree2)

    print("ESH:", judge_ESH, " SHK:", judge_SHK)

    if judge_ESH <= 10 and judge_SHK <= 10:
        cv2.putText(image, "Good!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif judge_ESH < 15 and judge_SHK < 15:
        cv2.putText(image, "Risk!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(image, "Not Good!!!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def judgeAngleFront(image, landmarks):
    # centerPoint
    ear_center = np.array([(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x +
                        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x) / 2,
                        (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y +
                        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y) / 2])

    shoulder_center = np.array([(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                       (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2])

    hip_center = np.array([(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x +
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                  (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2])

    knee_center = np.array([(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x +
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x) / 2,
                   (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y +
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2])
    
    # vectol
    vecESH_a = ear_center - shoulder_center
    vecESH_c = hip_center - shoulder_center
    vecSHK_a = shoulder_center - hip_center
    vecSHK_c = knee_center - hip_center

    # cosine
    cos_ESH = np.inner(vecESH_a, vecESH_c) / (np.linalg.norm(vecESH_a) * np.linalg.norm(vecESH_c))
    cos_SHK = np.inner(vecSHK_a, vecSHK_c) / (np.linalg.norm(vecSHK_a) * np.linalg.norm(vecSHK_c))

    # angle
    degree1 = np.rad2deg(np.arccos(cos_ESH))
    degree2 = np.rad2deg(np.arccos(cos_SHK))

    judge_ESH = abs(180 - degree1)
    judge_SHK = abs(180 - degree2)

    print("ESH:", judge_ESH, " SHK:", judge_SHK)

    if judge_ESH <= 15 and judge_SHK <= 15:
        cv2.putText(image, "Good!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif judge_ESH < 20 and judge_SHK < 20:
        cv2.putText(image, "Risk!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(image, "Not Good!!!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def drawCenterLine(image, landmarks):
    h, w, _ = image.shape

    # centerPoint
    ear_center = ((landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x +
                   landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x) / 2 * w,
                  (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y +
                   landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y) / 2 * h)

    shoulder_center = ((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2 * w,
                       (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2 * h)

    hip_center = ((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x +
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2 * w,
                  (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 * h)

    knee_center = ((landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x +
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x) / 2 * w,
                   (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y +
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2 * h)

    # ラインを描画
    cv2.line(image, (int(ear_center[0]), int(ear_center[1])),
             (int(shoulder_center[0]), int(shoulder_center[1])), (255, 0, 0), 3)

    cv2.line(image, (int(shoulder_center[0]), int(shoulder_center[1])),
             (int(hip_center[0]), int(hip_center[1])), (255, 0, 0), 3)

    cv2.line(image, (int(hip_center[0]), int(hip_center[1])),
             (int(knee_center[0]), int(knee_center[1])), (255, 0, 0), 3)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        # judge angle
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # x,y change 
        h, w, _ = image.shape
        lx, ly = int(shoulder.x * w), int(shoulder.y * h)
        rx, ry = int(right_shoulder.x * w), int(right_shoulder.y * h)

        # ユークリッド距離
        shoulder_distance = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
        
        if shoulder_distance < 50:
            cv2.putText(image, "SIDE", (0, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            judgeAngleSide(ear, shoulder, hip, knee, image)
        else:
            cv2.putText(image, "FRONT", (0, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            judgeAngleFront(image, landmarks)

        # draw centerLine
        drawCenterLine(image, landmarks)

    cv2.imshow("Posture Correction with Center Line", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()