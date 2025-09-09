import os
import mediapipe as mp
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

class ClsDrawCenterLine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def drawCenterLine(self, image, landmarks):
        h, w, _ = image.shape

        # centerPoint
        ear_center = ((landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x +
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x) / 2 * w,
                    (landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y +
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y) / 2 * h)

        shoulder_center = ((landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2 * w,
                        (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2 * h)

        hip_center = ((landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x +
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2 * w,
                    (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y +
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 * h)

        knee_center = ((landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x +
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x) / 2 * w,
                    (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y +
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2 * h)

        # draw line
        cv2.line(image, (int(ear_center[0]), int(ear_center[1])),
                (int(shoulder_center[0]), int(shoulder_center[1])), (255, 0, 0), 3)

        cv2.line(image, (int(shoulder_center[0]), int(shoulder_center[1])),
                (int(hip_center[0]), int(hip_center[1])), (255, 0, 0), 3)

        cv2.line(image, (int(hip_center[0]), int(hip_center[1])),
                (int(knee_center[0]), int(knee_center[1])), (255, 0, 0), 3)
