import cv2
import numpy as np


class FrameLoader:
    def __init__(self, config):
        self.config = config

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def write_video(self, frames, sequence, output_path):
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, self.config.FRAME_RATE, (width, height)
        )
        for frame_idx in sequence:
            out.write(frames[frame_idx])
        out.release()
