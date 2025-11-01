import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
from .config import Config


class SimilarityAnalyzer:
    def __init__(self, config):
        self.config = config

    def extract_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 36))
        return resized

    def calculate_similarity(self, frame1, frame2):
        feat1 = self.extract_features(frame1)
        feat2 = self.extract_features(frame2)

        score, _ = ssim(feat1, feat2, full=True)
        return score

    def compute_similarity_matrix(self, frames):
        n_frames = len(frames)
        similarity_matrix = np.zeros((n_frames, n_frames))
        features = [self.extract_features(frame) for frame in frames]

        def compute_pair(i, j):
            if i == j:
                return 1.0
            score, _ = ssim(features[i], features[j], full=True)
            return score

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = []
            for i in range(n_frames):
                for j in range(i, n_frames):
                    futures.append((i, j, executor.submit(compute_pair, i, j)))

            for i, j, future in futures:
                similarity = future.result()
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix
