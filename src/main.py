import time
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Remove the dots for absolute imports
from config import Config
from frame_loader import FrameLoader
from fast_reconstructor import FastReconstructor


def main():
    config = Config()

    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=== ULTRA-FAST VIDEO RECONSTRUCTION ===")
    print("Loading frames...")

    start_time = time.time()
    loader = FrameLoader(config)

    load_start = time.time()
    frames = loader.load_frames(config.INPUT_VIDEO)
    load_time = time.time() - load_start

    print(f"Loaded {len(frames)} frames in {load_time:.2f}s")

    if len(frames) != config.TOTAL_FRAMES:
        print(f"Warning: Expected {config.TOTAL_FRAMES} frames, got {len(frames)}")

    print("Reconstructing sequence...")
    reconstructor = FastReconstructor(config)

    recon_start = time.time()
    sequence = reconstructor.reconstruct_sequence(frames)
    recon_time = time.time() - recon_start

    print("Writing output video...")
    write_start = time.time()
    loader.write_video(frames, sequence, config.OUTPUT_VIDEO)
    write_time = time.time() - write_start

    total_time = time.time() - start_time

    print(f"\n=== EXECUTION SUMMARY ===")
    print(f"Frame loading: {load_time:.2f}s")
    print(f"Reconstruction: {recon_time:.2f}s")
    print(f"Video writing: {write_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Output: {config.OUTPUT_VIDEO}")

    with open("execution_time.log", "w") as f:
        f.write("ULTRA-FAST RECONSTRUCTION RESULTS\n")
        f.write("================================\n")
        f.write(f"Total Frames: {len(frames)}\n")
        f.write(f"Frame Rate: {config.FRAME_RATE}\n")
        f.write(f"Load Time: {load_time:.2f}s\n")
        f.write(f"Reconstruction Time: {recon_time:.2f}s\n")
        f.write(f"Write Time: {write_time:.2f}s\n")
        f.write(f"Total Time: {total_time:.2f}s\n")
        f.write(f"Frames Per Second: {len(frames) / total_time:.2f}\n")


if __name__ == "__main__":
    main()
