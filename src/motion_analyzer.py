import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class MotionAnalyzer:
    def __init__(self, config):
        self.config = config

    def compute_motion_score(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        if self.config.OPTICAL_FLOW_SCALE < 1.0:
            new_size = (
                int(gray1.shape[1] * self.config.OPTICAL_FLOW_SCALE),
                int(gray1.shape[0] * self.config.OPTICAL_FLOW_SCALE),
            )
            gray1 = cv2.resize(gray1, new_size)
            gray2 = cv2.resize(gray2, new_size)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 2, 10, 2, 3, 1.1, 0
        )
        motion_mag = np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))
        return motion_mag
