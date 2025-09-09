import os
import mediapipe as mp
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from ClsJudgeAngleFront import ClsJudgeAngleFront
from ClsJudgeAngleSide import ClsJudgeAngleSide
from ClsDrawCenterLine import ClsDrawCenterLine

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

def main():
    # instantiation
    clsJudgeAngleFront = ClsJudgeAngleFront()
    clsJudgeAngleSide = ClsJudgeAngleSide()
    clsDrawCenterLine = ClsDrawCenterLine()

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

            # Euclidean distance
            shoulder_distance = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
            
            if shoulder_distance < 50:
                cv2.putText(image, "SIDE", (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                clsJudgeAngleSide.judgeAngleSide(ear, shoulder, hip, knee, image)
            else:
                cv2.putText(image, "FRONT", (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                clsJudgeAngleFront.judgeAngleFront(image, landmarks)

            # draw centerLine
            clsDrawCenterLine.drawCenterLine(image, landmarks)

        cv2.imshow("Posture Correction with Center Line", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()