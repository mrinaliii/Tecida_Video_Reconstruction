import time
import cv2
from frame_extractor import FrameExtractor
from reconstructor import VideoReconstructor
from config import Config


def main():
    config = Config()

    print("Starting video frame reconstruction...")
    start_time = time.time()

    extractor = FrameExtractor(config)
    reconstructor = VideoReconstructor(config)

    print("Extracting frames from video...")
    frames = extractor.extract_frames(config.INPUT_VIDEO)

    print(f"Extracted {len(frames)} frames")

    print("Reconstructing video...")
    sequence = reconstructor.reconstruct_video(frames, config.OUTPUT_VIDEO)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Reconstruction completed in {execution_time:.2f} seconds")
    print(f"Output saved to: {config.OUTPUT_VIDEO}")

    with open("execution_time.log", "w") as f:
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"Frames Processed: {len(frames)}\n")
        f.write(f"Frame Rate: {config.FRAME_RATE}\n")


if __name__ == "__main__":
    main()
