import os
import mediapipe as mp
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

class ClsJudgeAngleSide:
    def judgeAngleSide(self, ear, shoulder, hip, knee, image):
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
            cv2.putText(image, "Good!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif judge_ESH < 15 and judge_SHK < 15:
            cv2.putText(image, "Risk!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(image, "Not Good!!!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)