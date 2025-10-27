import os


class Config:
    INPUT_VIDEO = "data/jumbled_video.mp4"
    OUTPUT_VIDEO = "output/reconstructed_video.mp4"
    FRAME_RATE = 30
    VIDEO_DURATION = 10
    TOTAL_FRAMES = FRAME_RATE * VIDEO_DURATION
    TEMP_DIR = "temp_frames"

    SIMILARITY_THRESHOLD = 0.85
    MAX_WORKERS = 8
