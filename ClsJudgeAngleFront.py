import os
import mediapipe as mp
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from ClsPlaySound import ClsPlaySound

class ClsJudgeAngleFront:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        clsPlaySound = ClsPlaySound()

    def judgeAngleFront(self, image, landmarks, screen_width, screen_height, frame_count):
        # centerPoint
        ear_center = np.array([(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x +
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x) / 2,
                            (landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y +
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y) / 2])

        shoulder_center = np.array([(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                        (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2])

        hip_center = np.array([(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x +
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                    (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y +
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2])

        knee_center = np.array([(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x +
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x) / 2,
                    (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y +
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2])
        
        h, w, _ = image.shape
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        lx, ly = int(left_shoulder.x * w), int(left_shoulder.y * h)
        rx, ry = int(right_shoulder.x * w), int(right_shoulder.y * h)

        diff = abs(ly - ry)

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

        if judge_ESH <= 15 and judge_SHK <= 15 and diff <= 5:
            if frame_count == 10:
                ClsPlaySound.playsound("Sound/Yes.mp3")
            cv2.putText(image, "Good!", (50, 100),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 4)
            cv2.putText(image, "Great posture!", (int(screen_width / 2)-100, screen_height-75),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 2)
        elif judge_ESH < 20 and judge_SHK < 20 and diff <= 15:
            cv2.putText(image, "Risk!", (50, 100),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 4)
            cv2.putText(image, "Be careful!! There is still hope!!", (int(screen_width / 2)-100, screen_height-75),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 2)
        else:
            if frame_count == 10:
                ClsPlaySound.playsound("Sound/No.mp3")
            cv2.putText(image, "Not Good!!!", (50, 100),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 4)
            cv2.putText(image, "I'm worried about your health...", (int(screen_width / 2)-100, screen_height-75),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)