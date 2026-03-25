"""
Dataset sample generator for The Shining film restoration.

Generates one folder per frame:
  {output}/shining_{id:06d}/
    scan.png          raw scan frame
    restored1.png     spatially aligned restored copy 1
    restored2.png     spatially aligned restored copy 2
    mask.png          binary defect mask

Optional (flags):
    corrected1.png    scan with restored-1 colour applied  [--save-corrected]
    corrected2.png    scan with restored-2 colour applied  [--save-corrected]
    inpainted.png     scan inpainted by LaMa (3-px dilated mask) [--save-inpainted]

Progress is saved to {output}/progress/:
    stats.jsonl           one JSON line per frame (load with pandas later)
    offset_r1.csv / offset_r2.csv   full anchor logs from the trackers
    alignment_{id:06d}.png          periodic alignment grid visualisations

Root-cause of confidence=0 in old logs
---------------------------------------
The short scan (shining_scan_short.mkv, 14 400 frames) ran out of frames at
~frame 14 800 so every subsequent search returned None → offset frozen,
confidence 0.  This script defaults to the full 35 mm scan (~206 k frames).

Usage
-----
python generate_masks.py --start 450 --count 400 --every 1 --output ./dataset
python generate_masks.py --start 0 --count 100000 --every 10 \\
    --adaptive-offset --save-corrected --save-inpainted --workers 6 \\
    --output ./dataset
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hardcoded defaults for The Shining
# ---------------------------------------------------------------------------
VIDEOS_DIR    = "/data/storage/users/priubrogent/filmdataset/videos"
SCAN_VIDEO    = f"{VIDEOS_DIR}/The.Shining.1980.35mm.Scan.FullScreen.HYBRID.OPEN.MATTE.1080p.mkv"
R1_VIDEO      = f"{VIDEOS_DIR}/shining_restored-46.mkv"
R1_OFFSET     = -46
R2_VIDEO      = f"{VIDEOS_DIR}/shining_restored_copy2-38.mp4"
R2_OFFSET     = -38
DATASET_PREFIX = "shining"
# ---------------------------------------------------------------------------

from video_io import extract_frame_with_offset
from alignment import align_three_images, generate_gradient_difference_mask, compute_defect_mask
from color_library import compute_color_matrix, apply_color_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dilate_mask(mask: np.ndarray, px: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, kernel)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Per-frame computation (runs in thread pool — no tracker calls here)
# ---------------------------------------------------------------------------

def compute_frame(
    frame_num:     int,
    scan_path:     str,
    r1_path:       str,  r1_offset: int,
    r2_path:       str,  r2_offset: int,
    threshold:     float,
    save_corrected: bool,
) -> dict:
    scan_bgr = extract_frame_with_offset(scan_path, frame_num, 0)
    if scan_bgr is None:
        return {"frame": frame_num, "failed": True, "reason": "scan_none"}

    r1_bgr_orig = extract_frame_with_offset(r1_path, frame_num, r1_offset)
    r2_bgr_orig = extract_frame_with_offset(r2_path, frame_num, r2_offset)
    if r1_bgr_orig is None or r2_bgr_orig is None:
        return {"frame": frame_num, "failed": True, "reason": "restored_none"}

    scan_f = scan_bgr.astype(np.float32) / 255.0
    r1_f   = r1_bgr_orig.astype(np.float32) / 255.0
    r2_f   = r2_bgr_orig.astype(np.float32) / 255.0

    r1_al, r2_al, M1, M2 = align_three_images(scan_f, r1_f, r2_f)
    if r1_al is None or r2_al is None:
        return {"frame": frame_num, "failed": True, "reason": "align_failed"}

    # Generate gradient difference maps (raw, not binarized)
    diff1 = generate_gradient_difference_mask(scan_f, r1_al)
    diff2 = generate_gradient_difference_mask(scan_f, r2_al)

    # Combine the two difference maps: take minimum (defect must appear in both)
    diff_combined = np.minimum(diff1, diff2)
    diff_combined = np.clip(diff_combined, 0, None)  # Only positive differences (scan has more edges)

    # Binary mask
    mask = compute_defect_mask(diff1, diff2, threshold=threshold)

    # Convert diff to uint8 for saving (normalize to 0-255)
    diff_max = max(diff_combined.max(), 1e-6)
    diff_uint8 = (np.clip(diff_combined / diff_max, 0, 1) * 255).astype(np.uint8)

    # Apply ECC warp directly to original uint8 frames — no float roundtrip, no color change
    h, w = scan_bgr.shape[:2]
    r1_base = r1_bgr_orig if r1_bgr_orig.shape[:2] == (h, w) else cv2.resize(r1_bgr_orig, (w, h), interpolation=cv2.INTER_LINEAR)
    r2_base = r2_bgr_orig if r2_bgr_orig.shape[:2] == (h, w) else cv2.resize(r2_bgr_orig, (w, h), interpolation=cv2.INTER_LINEAR)
    r1_bgr_al = cv2.warpAffine(r1_base, M1, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) if M1 is not None else r1_base.copy()
    r2_bgr_al = cv2.warpAffine(r2_base, M2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) if M2 is not None else r2_base.copy()

    # Unaligned originals (just resized to scan dims)
    r1_bgr_resized = r1_base
    r2_bgr_resized = r2_base

    out = {
        "frame":       frame_num,
        "failed":      False,
        "scan":        scan_bgr,
        "r1":          r1_bgr_al,          # Aligned restored1
        "r2":          r2_bgr_al,          # Aligned restored2
        "r1_orig":     r1_bgr_resized,     # Original restored1 (resized, not aligned)
        "r2_orig":     r2_bgr_resized,     # Original restored2 (resized, not aligned)
        "mask":        mask,               # Binary mask
        "mask_raw":    diff_uint8,         # Raw difference map (not binarized)
        "mask_pct":    float(np.mean(mask > 0) * 100),
        "r1_ecc":      M1 is not None,
        "r2_ecc":      M2 is not None,
    }

    if save_corrected:
        # Apply each restored's colour grade onto the scan frame
        # Result: scan content with restored colour grade (scan_corrected_1/2)
        scan_rgb = _bgr_to_rgb(scan_bgr)
        r1_rgb   = _bgr_to_rgb(r1_bgr_al)
        r2_rgb   = _bgr_to_rgb(r2_bgr_al)
        H1 = compute_color_matrix(reference=r1_rgb, source=scan_rgb)
        H2 = compute_color_matrix(reference=r2_rgb, source=scan_rgb)
        out["corrected1"] = _rgb_to_bgr(apply_color_matrix(scan_rgb, H1))
        out["corrected2"] = _rgb_to_bgr(apply_color_matrix(scan_rgb, H2))

    return out


# ---------------------------------------------------------------------------
# Save one frame's outputs
# ---------------------------------------------------------------------------

def save_frame(result: dict, output_root: str):
    """
    Save frame outputs into category-based folders:
        {output_root}/scan/{id}.png                 - Original scan (degraded)
        {output_root}/restored1/{id}.png            - Aligned restored copy 1
        {output_root}/restored2/{id}.png            - Aligned restored copy 2
        {output_root}/restored1_orig/{id}.png       - Original restored copy 1 (not aligned)
        {output_root}/restored2_orig/{id}.png       - Original restored copy 2 (not aligned)
        {output_root}/mask/{id}.png                 - Binary defect mask
        {output_root}/mask_raw/{id}.png             - Raw difference map (not binarized)
        {output_root}/scan_corrected_1/{id}.png     - Scan with restored1 colors (if --save-corrected)
        {output_root}/scan_corrected_2/{id}.png     - Scan with restored2 colors (if --save-corrected)
        {output_root}/inpainted/{id}.png            - Inpainted scan (if --save-inpainted)
    """
    fn = result["frame"]
    filename = f"{DATASET_PREFIX}_{fn:06d}.png"

    # Create category folders
    folders = ["scan", "restored1", "restored2", "restored1_orig", "restored2_orig", "mask", "mask_raw"]
    if "corrected1" in result:
        folders.extend(["scan_corrected_1", "scan_corrected_2"])
    if "inpainted" in result:
        folders.append("inpainted")

    for folder in folders:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    # Save images
    cv2.imwrite(os.path.join(output_root, "scan", filename),           result["scan"])
    cv2.imwrite(os.path.join(output_root, "restored1", filename),      result["r1"])
    cv2.imwrite(os.path.join(output_root, "restored2", filename),      result["r2"])
    cv2.imwrite(os.path.join(output_root, "restored1_orig", filename), result["r1_orig"])
    cv2.imwrite(os.path.join(output_root, "restored2_orig", filename), result["r2_orig"])
    cv2.imwrite(os.path.join(output_root, "mask", filename),           result["mask"])
    cv2.imwrite(os.path.join(output_root, "mask_raw", filename),       result["mask_raw"])

    if "corrected1" in result:
        cv2.imwrite(os.path.join(output_root, "scan_corrected_1", filename), result["corrected1"])
        cv2.imwrite(os.path.join(output_root, "scan_corrected_2", filename), result["corrected2"])

    if "inpainted" in result:
        cv2.imwrite(os.path.join(output_root, "inpainted", filename), result["inpainted"])


# ---------------------------------------------------------------------------
# Alignment visualisation grid
# ---------------------------------------------------------------------------

def make_alignment_vis(result: dict, r1_offset: int, r2_offset: int) -> np.ndarray:
    """4-panel strip: scan | restored1 | restored2 | mask  (all same width)."""
    target_w = 480
    h, w = result["scan"].shape[:2]
    target_h = int(h * target_w / w)

    def _resize(img):
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    mask_bgr = cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR)
    strip = np.hstack([
        _resize(result["scan"]),
        _resize(result["r1"]),
        _resize(result["r2"]),
        _resize(mask_bgr),
    ])

    # Add text labels
    labels = [
        "scan",
        f"restored1 (off={r1_offset:+d})",
        f"restored2 (off={r2_offset:+d})",
        f"mask ({result['mask_pct']:.1f}% defect)",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        x = i * target_w + 6
        cv2.putText(strip, label, (x, 22), font, 0.55, (0, 0, 0),   2, cv2.LINE_AA)
        cv2.putText(strip, label, (x, 22), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return strip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate The Shining dataset samples (scan + aligned restored + masks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Video overrides (defaults to hardcoded Shining paths)
    parser.add_argument("--scan",      default=SCAN_VIDEO, help=f"Scan video (default: {SCAN_VIDEO})")
    parser.add_argument("--r1",        default=R1_VIDEO,   help=f"Restored-1 video (default: {R1_VIDEO})")
    parser.add_argument("--r1-offset", type=int, default=R1_OFFSET, help=f"Restored-1 initial offset (default: {R1_OFFSET})")
    parser.add_argument("--r2",        default=R2_VIDEO,   help=f"Restored-2 video (default: {R2_VIDEO})")
    parser.add_argument("--r2-offset", type=int, default=R2_OFFSET, help=f"Restored-2 initial offset (default: {R2_OFFSET})")

    # Frame selection
    parser.add_argument("--start",  type=int, default=0,      help="First frame number (default: 0)")
    parser.add_argument("--count",  type=int, default=400,    help="Frame span (default: 400)")
    parser.add_argument("--every",  type=int, default=1,      help="Process every N frames (default: 1)")

    # Processing
    parser.add_argument("--threshold", type=float, default=0.10, help="Gradient-diff threshold (default: 0.10)")
    parser.add_argument("--workers",   type=int, default=max(1, os.cpu_count() - 1),
                        help="Thread-pool workers for frame alignment (default: cpu_count-1)")
    parser.add_argument("--output",    default="./dataset", help="Root output directory")

    # Optional outputs
    parser.add_argument("--save-corrected", action="store_true",
                        help="Save scan with each restored's colour transferred (corrected1/2.png)")
    parser.add_argument("--save-inpainted", action="store_true",
                        help="Save LaMa-inpainted scan (3-px dilated mask).  Requires simple-lama-inpainting.")

    # Visualisation
    parser.add_argument("--vis-every", type=int, default=100, metavar="N",
                        help="Save alignment grid PNG every N processed frames (default: 100)")

    # Adaptive offset
    adapt = parser.add_argument_group("Adaptive offset (temporal drift correction)")
    adapt.add_argument("--adaptive-offset", action="store_true",
                       help="Enable drift-correcting temporal offset tracker.")
    adapt.add_argument("--anchor-interval", type=int, default=50,
                       help="Re-check offset every N frames (default: 50)")
    adapt.add_argument("--search-range",    type=int, default=8,
                       help="Initial search half-width in frames (default: 8)")

    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    progress_dir = os.path.join(args.output, "progress")
    os.makedirs(progress_dir, exist_ok=True)

    frame_numbers = list(range(args.start, args.start + args.count, args.every))
    total = len(frame_numbers)

    print(f"\n{'='*60}")
    print(f"  Scan:       {args.scan}")
    print(f"  Restored-1: {args.r1}  (offset {args.r1_offset:+d})")
    print(f"  Restored-2: {args.r2}  (offset {args.r2_offset:+d})")
    print(f"  Frames:     {total}  ({args.start}..{args.start+args.count-1}, step {args.every})")
    print(f"  Workers:    {args.workers}")
    print(f"  Output:     {args.output}")
    print(f"  Adaptive:   {args.adaptive_offset}")
    print(f"  Corrected:  {args.save_corrected}")
    print(f"  Inpainted:  {args.save_inpainted}")
    print(f"{'='*60}\n")

    # ── Adaptive trackers ────────────────────────────────────────────────────
    trackers = None
    if args.adaptive_offset:
        from temporal_offset_tracker import TemporalOffsetTracker
        trackers = [
            TemporalOffsetTracker(args.scan, args.r1, args.r1_offset,
                                  anchor_interval=args.anchor_interval,
                                  search_range=args.search_range, verbose=True),
            TemporalOffsetTracker(args.scan, args.r2, args.r2_offset,
                                  anchor_interval=args.anchor_interval,
                                  search_range=args.search_range, verbose=True),
        ]

    # ── LaMa ────────────────────────────────────────────────────────────────
    lama = None
    if args.save_inpainted:
        try:
            sys.path.insert(0, str(Path(__file__).parent / "inpainting"))
            from lama_inpainting import LamaInpainter
            lama = LamaInpainter(device="cpu", backend="simple-lama")
            print("LaMa loaded.\n")
        except Exception as e:
            print(f"[WARNING] LaMa not available ({e}). --save-inpainted will be skipped.\n")
            args.save_inpainted = False

    # ── Stats file ───────────────────────────────────────────────────────────
    stats_path = os.path.join(progress_dir, "stats.jsonl")
    stats_file = open(stats_path, "w")

    # ── Thread-pool pipeline ─────────────────────────────────────────────────
    def _get_offsets(fn):
        if trackers:
            return trackers[0].get_offset(fn), trackers[1].get_offset(fn)
        return args.r1_offset, args.r2_offset

    def _handle(result: dict, r1_off: int, r2_off: int, processed_count: list):
        fn = result["frame"]

        if result["failed"]:
            print(f"  [SKIP] frame {fn}: {result['reason']}")
            stats_file.write(json.dumps({
                "frame": fn, "failed": True, "reason": result["reason"],
                "r1_offset": r1_off, "r2_offset": r2_off,
            }) + "\n")
            stats_file.flush()
            return

        # LaMa inpainting (serial, in main thread)
        if args.save_inpainted and lama is not None:
            from PIL import Image
            dilated = _dilate_mask(result["mask"], px=3)
            scan_rgb_pil = Image.fromarray(_bgr_to_rgb(result["scan"]))
            mask_pil     = Image.fromarray(dilated)
            inpainted    = lama.inpaint(scan_rgb_pil, mask_pil)
            result["inpainted"] = _rgb_to_bgr(np.array(inpainted))

        save_frame(result, args.output)

        # Stats
        row = {
            "frame":    fn,
            "failed":   False,
            "r1_offset": r1_off,
            "r2_offset": r2_off,
            "mask_pct": round(result["mask_pct"], 4),
            "r1_ecc":   result["r1_ecc"],
            "r2_ecc":   result["r2_ecc"],
        }
        stats_file.write(json.dumps(row) + "\n")
        stats_file.flush()

        # Periodic alignment visualisation
        n = processed_count[0]
        processed_count[0] += 1
        if n % args.vis_every == 0:
            vis = make_alignment_vis(result, r1_off, r2_off)
            vis_path = os.path.join(progress_dir, f"alignment_{fn:06d}.png")
            cv2.imwrite(vis_path, vis)
            print(f"  [VIS] alignment grid → {vis_path}")

    processed_count = [0]

    frame_queue = deque(frame_numbers)
    pending = {}      # future → (fn, r1_off, r2_off)
    window  = args.workers * 2

    pbar = tqdm(total=total, desc="Generating dataset", unit="frame")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:

        def _enqueue(fn):
            r1_off, r2_off = _get_offsets(fn)
            fut = pool.submit(
                compute_frame,
                fn, args.scan,
                args.r1, r1_off,
                args.r2, r2_off,
                args.threshold,
                args.save_corrected,
            )
            pending[fut] = (fn, r1_off, r2_off)

        # Prime the window
        while frame_queue and len(pending) < window:
            fn = frame_queue.popleft()
            _enqueue(fn)

        while pending:
            done_set, _ = wait(list(pending), timeout=2.0, return_when=FIRST_COMPLETED)

            for fut in done_set:
                fn, r1_off, r2_off = pending.pop(fut)
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"  [ERROR] frame {fn}: {e}")
                    result = {"frame": fn, "failed": True, "reason": str(e)}

                _handle(result, r1_off, r2_off, processed_count)
                pbar.update(1)

                # Check if scan exhausted (trackers all dead)
                if trackers and all(t.is_exhausted for t in trackers):
                    print("\n[INFO] Scan exhausted — all trackers dead. Stopping early.")
                    frame_queue.clear()
                    break

                # Enqueue next frame
                if frame_queue:
                    _enqueue(frame_queue.popleft())

    pbar.close()
    stats_file.close()

    # ── Save offset logs ─────────────────────────────────────────────────────
    if trackers:
        trackers[0].save_log(os.path.join(progress_dir, "offset_r1.csv"))
        trackers[1].save_log(os.path.join(progress_dir, "offset_r2.csv"))

    done_n = processed_count[0]
    print(f"\nDone. {done_n}/{total} frames saved → '{args.output}'")
    print(f"Progress data → '{progress_dir}'")


if __name__ == "__main__":
    main()
