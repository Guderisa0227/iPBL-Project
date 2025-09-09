import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import mediapipe as mp

class ClsResizeWindow:  
    def resizeWindow(self, image, screen_width, screen_height):
        size=(screen_width, screen_height)

        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))

        top = (size[1] - new_h) // 2
        bottom = size[1] - new_h - top
        left = (size[0] - new_w) // 2
        right = size[0] - new_w - left

        color = [0, 0, 0]
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded_image

    def setFullscreen(self, window_name):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
