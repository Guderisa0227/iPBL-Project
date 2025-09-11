import os
import mediapipe as mp
import numpy as np
import tkinter as tk

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from ClsJudgeAngleFront import ClsJudgeAngleFront
from ClsJudgeAngleSide import ClsJudgeAngleSide
from ClsDrawCenterLine import ClsDrawCenterLine
from ClsResizeWindow import ClsResizeWindow

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
    clsResizeWindow = ClsResizeWindow()

    cap = cv2.VideoCapture(0)
    frame_count = 0
    N = 20

    # window name
    window_name = "Camera Fullscreen"

    # create window
    clsResizeWindow.setFullscreen(window_name)

    # change fullscreen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = clsResizeWindow.resizeWindow(frame, screen_width, screen_height)
        frame_count += 1

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
                cv2.putText(image, "SIDE", (50, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 0), 2)
                clsJudgeAngleSide.judgeAngleSide(ear, shoulder, hip, knee, image, screen_width, screen_height, frame_count)
            else:
                cv2.putText(image, "FRONT", (50, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 0, 255), 2)
                clsJudgeAngleFront.judgeAngleFront(image, landmarks, screen_width, screen_height, frame_count)

            # draw centerLine
            clsDrawCenterLine.drawCenterLine(image, landmarks)

            if frame_count >= N:
                frame_count = 0

        cv2.imshow(window_name, image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    root.destroy()
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()