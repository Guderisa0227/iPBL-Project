import os
import mediapipe as mp
import numpy as np
import time
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from ClsPlaySound import ClsPlaySound

class ClsJudgeAngleSide:
    #kp start
    def __init__(self):
        self.clsPlaySound = ClsPlaySound()
        self.bad_posture_start = None
        self.was_bad_posture = False
    #kp end
    def judgeAngleSide(self, ear, shoulder, hip, knee, image, screen_width, screen_height, frame_count):
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
            #kp start
            if frame_count == 10:
                if self.was_bad_posture:  # If previously was in bad posture
                    self.clsPlaySound.play_with_delay("Sound/Yes.mp3", delay=0)  # Play immediately
                    self.was_bad_posture = False
                self.bad_posture_start = None
            #kp end
            (width, height), baseline= cv2.getTextSize("Good!", cv2.FONT_HERSHEY_TRIPLEX, 2, 4)
            top_left_point = (25, 125 - height)
            bottom_right_point = (75 + width, 175)
            cv2.rectangle(image, top_left_point, bottom_right_point, (0,0,0), -1)
            cv2.putText(image, "Good!", (50, 150),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 4)
            
            (width, height), baseline= cv2.getTextSize("Great posture!", cv2.FONT_HERSHEY_TRIPLEX, 1.5, 2)
            top_left_point = (int(screen_width / 2)-120, screen_height-100 - height)
            bottom_right_point = (int(screen_width / 2)-80 + width, screen_height-50)
            cv2.rectangle(image, top_left_point, bottom_right_point, (0,0,0), -1)
            cv2.putText(image, "Great posture!", (int(screen_width/2)-100, screen_height-75),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 2)
        elif judge_ESH < 15 and judge_SHK < 15:
            (width, height), baseline= cv2.getTextSize("Risk!", cv2.FONT_HERSHEY_TRIPLEX, 2, 4)
            top_left_point = (25, 125 - height)
            bottom_right_point = (75 + width, 175)
            cv2.rectangle(image, top_left_point, bottom_right_point, (0,0,0), -1)
            cv2.putText(image, "Risk!", (50, 150),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 4)
            
            (width, height), baseline= cv2.getTextSize("Be careful!! There is still hope!!", cv2.FONT_HERSHEY_TRIPLEX, 1.5, 2)
            top_left_point = (int(screen_width / 2)-120, screen_height-100 - height)
            bottom_right_point = (int(screen_width / 2)-80 + width, screen_height-50)
            cv2.rectangle(image, top_left_point, bottom_right_point, (0,0,0), -1)
            cv2.putText(image, "Be careful!! There is still hope!!", (int(screen_width / 2)-100, screen_height-75),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 2)
        else:
            #KP start
            current_time = time.time()
            if self.bad_posture_start is None:
                self.bad_posture_start = current_time
                self.was_bad_posture = True
            elif current_time - self.bad_posture_start >= 3:  # 3 seconds threshold
                self.clsPlaySound.play_with_delay("Sound/No.mp3", delay=0)
                self.bad_posture_start = current_time  # Reset timer
            #KP end
            (width, height), baseline= cv2.getTextSize("Not Good!!!", cv2.FONT_HERSHEY_TRIPLEX, 2, 4)
            top_left_point = (25, 125 - height)
            bottom_right_point = (75 + width, 175)
            cv2.rectangle(image, top_left_point, bottom_right_point, (0,0,0), -1)
            cv2.putText(image, "Not Good!!!", (50, 150),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 4)
            
            (width, height), baseline= cv2.getTextSize("I'm worried about your health...", cv2.FONT_HERSHEY_TRIPLEX, 1.5, 2)
            top_left_point = (int(screen_width / 2)-120, screen_height-100 - height)
            bottom_right_point = (int(screen_width / 2)-80 + width, screen_height-50)
            cv2.rectangle(image, top_left_point, bottom_right_point, (0,0,0), -1)
            cv2.putText(image, "I'm worried about your health...", (int(screen_width / 2)-100, screen_height-75),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)