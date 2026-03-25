"""
Video I/O utilities for film frame extraction.
"""

import cv2
import numpy as np


def extract_frame_with_offset(video_path: str, frame_number: int, offset: int = 0) -> np.ndarray | None:
    """
    Extract a frame from a video at (frame_number + offset).

    Args:
        video_path:    Path to the video file.
        frame_number:  The logical frame number (e.g. same across all video versions).
        offset:        Temporal offset for this video version (can be negative).
                       E.g. offset=-38 means this copy starts 38 frames earlier.

    Returns:
        Frame as a uint8 BGR numpy array, or None on failure.
    """
    target_frame = frame_number + offset

    if target_frame < 0:
        print(f"Error: target frame {target_frame} is negative (frame={frame_number}, offset={offset})")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video '{video_path}'")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_frame >= total_frames:
        print(f"Error: target frame {target_frame} exceeds total frames {total_frames} in '{video_path}'")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: could not read frame {target_frame} from '{video_path}'")
        return None

    return frame
