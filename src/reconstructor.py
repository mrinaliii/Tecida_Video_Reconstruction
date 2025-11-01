import numpy as np
import cv2
from similarity_analyzer import SimilarityAnalyzer


class VideoReconstructor:
    def __init__(self, config):
        self.config = config
        self.similarity_analyzer = SimilarityAnalyzer(config)

    def find_optimal_sequence_greedy(self, frames, similarity_matrix):
        n_frames = len(frames)
        used_frames = set()
        sequence = []

        current_frame = np.argmax(np.sum(similarity_matrix, axis=1))
        sequence.append(current_frame)
        used_frames.add(current_frame)

        for _ in range(n_frames - 1):
            best_next = -1
            best_similarity = -1

            for next_frame in range(n_frames):
                if next_frame in used_frames:
                    continue

                similarity = similarity_matrix[current_frame, next_frame]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_next = next_frame

            if best_next != -1:
                sequence.append(best_next)
                used_frames.add(best_next)
                current_frame = best_next

        return sequence

    def find_optimal_sequence_mst(self, frames, similarity_matrix):
        n_frames = len(frames)
        distance_matrix = 1 - similarity_matrix

        in_mst = [False] * n_frames
        key = [float("inf")] * n_frames
        parent = [-1] * n_frames

        key[0] = 0

        for _ in range(n_frames):
            min_key = float("inf")
            u = -1

            for i in range(n_frames):
                if not in_mst[i] and key[i] < min_key:
                    min_key = key[i]
                    u = i

            in_mst[u] = True

            for v in range(n_frames):
                if (
                    not in_mst[v]
                    and distance_matrix[u, v] < key[v]
                    and u != v
                    and distance_matrix[u, v] > 0
                ):
                    key[v] = distance_matrix[u, v]
                    parent[v] = u

        children = [[] for _ in range(n_frames)]
        for i in range(1, n_frames):
            children[parent[i]].append(i)

        sequence = []
        stack = [0]

        while stack:
            node = stack.pop()
            sequence.append(node)
            stack.extend(reversed(children[node]))

        return sequence

    def reconstruct_video(self, frames, output_path):
        print("Computing similarity matrix...")
        similarity_matrix = self.similarity_analyzer.compute_similarity_matrix(frames)

        print("Finding optimal sequence...")
        sequence = self.find_optimal_sequence_mst(frames, similarity_matrix)

        print("Writing reconstructed video...")
        self.write_video(frames, sequence, output_path)

        return sequence

    def write_video(self, frames, sequence, output_path):
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, self.config.FRAME_RATE, (width, height)
        )

        for frame_idx in sequence:
            out.write(frames[frame_idx])

        out.release()
