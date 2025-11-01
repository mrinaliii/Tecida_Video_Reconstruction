import numpy as np
import cv2
from sklearn.cluster import KMeans
from .similarity_analyzer import SimilarityAnalyzer


class AdvancedVideoReconstructor:
    def __init__(self, config):
        self.config = config
        self.similarity_analyzer = SimilarityAnalyzer(config)

    def optical_flow_sequence(self, frames):
        n_frames = len(frames)
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

        flow_costs = np.zeros((n_frames, n_frames))

        for i in range(n_frames):
            for j in range(n_frames):
                if i != j:
                    flow = cv2.calcOpticalFlowFarneback(
                        gray_frames[i], gray_frames[j], None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    flow_costs[i, j] = np.mean(np.abs(flow))

        return self.solve_tsp(flow_costs)

    def solve_tsp(self, cost_matrix):
        n = len(cost_matrix)
        unvisited = set(range(n))
        path = [np.argmin(np.sum(cost_matrix, axis=1))]
        unvisited.remove(path[0])

        while unvisited:
            current = path[-1]
            best_next = min(unvisited, key=lambda x: cost_matrix[current, x])
            path.append(best_next)
            unvisited.remove(best_next)

        return path

    def hybrid_reconstruction(self, frames):
        similarity_matrix = self.similarity_analyzer.compute_similarity_matrix(frames)

        seq1 = self.find_optimal_sequence_mst(frames, similarity_matrix)

        distance_matrix = 1 - similarity_matrix
        seq2 = self.solve_tsp(distance_matrix)

        return self.merge_sequences(seq1, seq2, frames)

    def find_optimal_sequence_mst(self, frames, similarity_matrix):
        n_frames = len(frames)
        distance_matrix = 1 - similarity_matrix

        in_mst = [False] * n_frames
        key = [float("inf")] * n_frames
        parent = [-1] * n_frames

        key[0] = 0

        for _ in range(n_frames):
            u = min((key[i], i) for i in range(n_frames) if not in_mst[i])[1]
            in_mst[u] = True

            for v in range(n_frames):
                if not in_mst[v] and distance_matrix[u, v] < key[v]:
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

    def merge_sequences(self, seq1, seq2, frames):
        analyzer = SimilarityAnalyzer(self.config)

        def sequence_quality(sequence):
            total_similarity = 0
            for i in range(len(sequence) - 1):
                total_similarity += analyzer.calculate_similarity(
                    frames[sequence[i]], frames[sequence[i + 1]]
                )
            return total_similarity

        quality1 = sequence_quality(seq1)
        quality2 = sequence_quality(seq2)

        return seq1 if quality1 > quality2 else seq2
