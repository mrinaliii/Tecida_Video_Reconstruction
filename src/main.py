import time
import argparse
import cv2
from frame_extractor import FrameExtractor
from reconstructor import VideoReconstructor
from advanced_reconstructor import AdvancedVideoReconstructor
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Video Frame Reconstruction")
    parser.add_argument(
        "--input", default="data/jumbled_video.mp4", help="Input video path"
    )
    parser.add_argument(
        "--output", default="output/reconstructed_video.mp4", help="Output video path"
    )
    parser.add_argument(
        "--method",
        choices=["basic", "advanced"],
        default="advanced",
        help="Reconstruction method",
    )

    args = parser.parse_args()

    config = Config()
    config.INPUT_VIDEO = args.input
    config.OUTPUT_VIDEO = args.output

    print("Starting video frame reconstruction...")
    start_time = time.time()

    extractor = FrameExtractor(config)

    print("Extracting frames from video...")
    frames = extractor.extract_frames(config.INPUT_VIDEO)
    print(f"Extracted {len(frames)} frames")

    if args.method == "basic":
        reconstructor = VideoReconstructor(config)
    else:
        reconstructor = AdvancedVideoReconstructor(config)

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
        f.write(f"Method: {args.method}\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Output: {args.output}\n")


if __name__ == "__main__":
    main()
