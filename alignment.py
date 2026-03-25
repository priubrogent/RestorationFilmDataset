"""
Spatial alignment and defect mask generation.

All images are expected as float32 BGR in [0, 1] range unless stated otherwise.
"""

import cv2
import numpy as np


def align_two_images(tgt: np.ndarray, src: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Align src to tgt using ECC (Enhanced Correlation Coefficient) with an affine motion model.

    Args:
        tgt: Target / reference image (float32 BGR, 0–1).
        src: Source image to warp onto tgt (float32 BGR, 0–1).
             Will be resized to tgt's dimensions if different.

    Returns:
        (aligned, M):
            aligned – warped src as float32 BGR, same size as tgt. None on failure.
            M       – 2×3 affine transformation matrix. None if ECC failed or confidence low.
    """
    h, w = tgt.shape[:2]

    # Convert to uint8 for ECC computation (grayscale)
    tgt_gray = cv2.cvtColor((tgt * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Resize src to target dimensions first (using high-quality Lanczos)
    if src.shape[:2] != tgt.shape[:2]:
        src_resized = cv2.resize(src, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        src_resized = src.copy()

    src_gray = cv2.cvtColor((src_resized * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    try:
        cc, M = cv2.findTransformECC(src_gray, tgt_gray, M, cv2.MOTION_AFFINE, criteria, None, 1)
    except cv2.error as e:
        print(f"ECC alignment failed: {e}")
        return None, None

    if cc < 0.5:
        # Low confidence – return a plain resize instead of a bad warp
        return src_resized, None

    aligned = cv2.warpAffine(src_resized, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    return aligned, M


def align_three_images(
    ref: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Align img2 and img3 independently to ref.

    Returns:
        (img2_aligned, img3_aligned, M2, M3)
    """
    img2_aligned, M2 = align_two_images(ref, img2)
    img3_aligned, M3 = align_two_images(ref, img3)
    return img2_aligned, img3_aligned, M2, M3


def generate_gradient_difference_mask(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute the signed Sobel-gradient magnitude difference between two images.

    Returns a float32 map (same H×W as inputs) where large positive values indicate
    edges present in img1 but absent in img2 — i.e. likely defect regions in img1.

    Args:
        img1: Reference image (float32 BGR, 0–1) — typically the film scan.
        img2: Comparison image (float32 BGR, 0–1) — typically an aligned restored copy.

    Returns:
        diff_map: float32 H×W array. Range roughly [−1, 1].
    """
    def _grad_mag(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    return _grad_mag(img1) - _grad_mag(img2)


def compute_defect_mask(
    diff_scan_vs_restored1: np.ndarray,
    diff_scan_vs_restored2: np.ndarray,
    threshold: float = 0.10,
) -> np.ndarray:
    """
    Combine two gradient-difference maps into a binary defect mask.

    A pixel is marked as a defect only if it exceeds the threshold in BOTH
    comparisons (scan vs restored1 AND scan vs restored2), reducing false positives.

    Args:
        diff_scan_vs_restored1: Output of generate_gradient_difference_mask(scan, restored1_aligned).
        diff_scan_vs_restored2: Output of generate_gradient_difference_mask(scan, restored2_aligned).
        threshold: Minimum gradient difference to flag as defect.

    Returns:
        Binary uint8 mask (255 = defect, 0 = clean), same H×W as inputs.
    """
    mask = ((diff_scan_vs_restored1 > threshold) & (diff_scan_vs_restored2 > threshold))
    return mask.astype(np.uint8) * 255
