import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from motion_analyzer import MotionAnalyzer


class FastReconstructor:
    def __init__(self, config):
        self.config = config
        self.motion_analyzer = MotionAnalyzer(config)

    def find_start_frame(self, frames):
        brightness = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(np.mean(gray))
        return np.argsort(brightness)[len(brightness) // 2]

    def reconstruct_sequence(self, frames):
        n = len(frames)
        start_idx = self.find_start_frame(frames)

        used = set([start_idx])
        sequence = [start_idx]
        current_idx = start_idx

        for step in range(n - 1):
            if step % 20 == 0:
                print(f"Processing frame {step + 1}/{n - 1}")

            best_next = -1
            best_motion = float("inf")

            candidates = self.select_candidates(frames, current_idx, used, n)

            motion_scores = []
            with ThreadPoolExecutor(
                max_workers=min(self.config.MAX_WORKERS, len(candidates))
            ) as executor:
                futures = []
                for cand_idx in candidates:
                    future = executor.submit(
                        self.motion_analyzer.compute_motion_score,
                        frames[current_idx],
                        frames[cand_idx],
                    )
                    futures.append((cand_idx, future))

                for cand_idx, future in futures:
                    motion = future.result()
                    motion_scores.append((cand_idx, motion))
                    if motion < best_motion:
                        best_motion = motion
                        best_next = cand_idx

            if best_next == -1 and candidates:
                best_next = candidates[0]

            if best_next != -1:
                sequence.append(best_next)
                used.add(best_next)
                current_idx = best_next

        return sequence

    def select_candidates(self, frames, current_idx, used, total_frames):
        unused = [i for i in range(total_frames) if i not in used]

        if len(unused) <= 20:
            return unused

        return np.random.choice(unused, 20, replace=False).tolist()
