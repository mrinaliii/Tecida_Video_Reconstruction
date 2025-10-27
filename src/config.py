import os


class Config:
    INPUT_VIDEO = "data/jumbled_video.mp4"
    OUTPUT_VIDEO = "output/reconstructed_video.mp4"
    FRAME_RATE = 30
    VIDEO_DURATION = 10
    TOTAL_FRAMES = 300
    MAX_WORKERS = 10
    OPTICAL_FLOW_SCALE = 0.5
