import cv2
import os
from tqdm import tqdm
from config import Config


class FrameExtractor:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.TEMP_DIR, exist_ok=True)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1

        cap.release()
        return frames

    def save_frames(self, frames, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(f"{output_dir}/frame_{i:04d}.jpg", frame)

    def load_frames(self, input_dir):
        frames = []
        frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

        for frame_file in frame_files:
            frame_path = os.path.join(input_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)

        return frames
