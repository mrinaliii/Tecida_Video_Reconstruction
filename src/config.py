import os


class Config:
    INPUT_VIDEO = "data/jumbled_video.mp4"
    OUTPUT_VIDEO = "output/reconstructed_video.mp4"
    FRAME_RATE = 30
    VIDEO_DURATION = 10
    TOTAL_FRAMES = 300
    MAX_WORKERS = 12  # trying optimization
    OPTICAL_FLOW_SCALE = 0.3  # reduced for speed
    CANDIDATE_POOL_SIZE = 15  # reduced candidate search
