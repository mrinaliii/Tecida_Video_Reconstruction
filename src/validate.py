import cv2
import numpy as np


def validate_video(file_path, expected_frames=300, expected_fps=30):
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        return False, "Cannot open video file"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    valid = True
    messages = []

    if frame_count != expected_frames:
        valid = False
        messages.append(f"Frame count: {frame_count} (expected {expected_frames})")

    if abs(fps - expected_fps) > 1:
        valid = False
        messages.append(f"FPS: {fps:.1f} (expected {expected_fps})")

    if width < 1280 or height < 720:
        messages.append(f"Resolution: {width}x{height} (may be low quality)")

    return valid, messages


if __name__ == "__main__":
    config = Config()

    print("Validating input video...")
    valid, messages = validate_video(config.INPUT_VIDEO)

    if valid:
        print("✓ Input video meets specifications")
    else:
        print("✗ Input video issues:")
        for msg in messages:
            print(f"  - {msg}")

    for msg in messages:
        if "may be" in msg:
            print(f"Note: {msg}")
