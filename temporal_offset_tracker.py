"""
Adaptive temporal offset tracker for film video alignment.

Problem
-------
Different versions of the same film (scan vs restored) have a temporal offset
that is NOT constant — it drifts slowly over the duration of the film due to
slight encoding speed differences, dropped frames, or reel splices.

Solution
--------
Every `anchor_interval` frames we do a fast local search: extract the scan frame
at the current position, then extract `2*search_range + 1` candidate frames from
the restored video (centred on the last known offset) and pick the one with the
highest normalised cross-correlation (NCC). Between anchors we reuse the last
confirmed offset — drift is slow so this is fine.

The search is done on low-resolution grayscale frames (~240×135) so it adds
negligible overhead.

OPTIMIZED version:
- Batch extraction: All candidate restored frames are read in a single sequential
  pass through the video, avoiding costly random seeks.
- Cached VideoCapture: caps opened once in __init__, never closed until __del__.
- Single-frame matching by default (n_verify=0) for speed. Multi-frame optional.
- Smaller default search_range (8) since drift is typically slow.
- Conservative offset updates: only update if new offset is within max_jump of
  current offset, preventing wild jumps from false matches.
"""

import csv
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path


def _to_small_gray(frame: np.ndarray, resize: tuple) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, resize, interpolation=cv2.INTER_AREA).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom >= 1e-8 else 0.0


def _read_frame_sequential(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Read next frame without seeking (fast sequential access)."""
    ret, frame = cap.read()
    return frame if ret else None


def _seek_and_read(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """Seek to specific frame and read it."""
    if frame_idx < 0:
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def _batch_read_frames(cap: cv2.VideoCapture, start_idx: int, count: int) -> list[tuple[int, np.ndarray | None]]:
    """
    Read `count` consecutive frames starting at `start_idx`.
    Returns list of (frame_idx, frame_or_None) tuples.
    Much faster than random seeking for each frame.
    """
    results = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_idx < 0:
        # Handle negative start: pad with None
        for i in range(start_idx, min(0, start_idx + count)):
            results.append((i, None))
        start_idx = 0
        count = count - len(results)

    if start_idx >= total or count <= 0:
        return results

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    for i in range(count):
        idx = start_idx + i
        if idx >= total:
            results.append((idx, None))
        else:
            ret, frame = cap.read()
            results.append((idx, frame if ret else None))

    return results


@dataclass
class AnchorPoint:
    frame_num:  int
    offset:     int
    confidence: float
    searched:   bool


class TemporalOffsetTracker:
    """
    Tracks temporal offset drift between scan and restored video.

    The offset convention is: restored_frame = scan_frame + offset
    So if offset=-38, scan frame 400 corresponds to restored frame 362.
    """
    def __init__(
        self,
        scan_path: str,
        restored_path: str,
        initial_offset: int,
        anchor_interval: int = 50,
        search_range: int = 8,
        resize: tuple = (320, 180),
        verbose: bool = True,
        min_conf: float = 0.70,
        keep_threshold: float = 0.60,
        max_jump: int = 5,
    ):
        """
        Args:
            scan_path: Path to the scan (degraded) video
            restored_path: Path to the restored video
            initial_offset: Known starting offset (restored = scan + offset)
            anchor_interval: Re-check offset every N frames
            search_range: Search ±N frames around current offset
            resize: Thumbnail size for NCC comparison
            verbose: Print debug info
            min_conf: Minimum NCC to accept a match
            keep_threshold: Below this NCC, keep previous offset
            max_jump: Maximum allowed offset change per anchor (prevents wild jumps)
        """
        self.scan_path       = scan_path
        self.restored_path   = restored_path
        self.anchor_interval = anchor_interval
        self.search_range    = search_range
        self.resize          = resize
        self.verbose         = verbose
        self.min_conf        = min_conf
        self.keep_threshold  = keep_threshold
        self.max_jump        = max_jump

        self._initial_offset     = initial_offset
        self._current_offset     = initial_offset
        self._consecutive_none   = 0
        self._anchors: list[AnchorPoint] = [
            AnchorPoint(frame_num=0, offset=initial_offset, confidence=1.0, searched=False)
        ]

        self._scan_cap     = cv2.VideoCapture(scan_path)
        self._restored_cap = cv2.VideoCapture(restored_path)
        if not self._scan_cap.isOpened():
            raise RuntimeError(f"Cannot open scan video: {scan_path}")
        if not self._restored_cap.isOpened():
            raise RuntimeError(f"Cannot open restored video: {restored_path}")

    def __del__(self):
        try:
            self._scan_cap.release()
            self._restored_cap.release()
        except Exception:
            pass

    @property
    def is_exhausted(self) -> bool:
        return self._consecutive_none >= 3

    def get_offset(self, frame_num: int) -> int:
        """Get the offset for a given scan frame number."""
        if frame_num % self.anchor_interval == 0:
            new_offset, confidence = self._search_best_offset(frame_num)

            if confidence == -1.0:
                self._consecutive_none += 1
                confidence = 0.0
            else:
                self._consecutive_none = 0

            # Decide whether to update offset
            jump = abs(new_offset - self._current_offset)
            if confidence >= self.keep_threshold and jump <= self.max_jump:
                self._current_offset = new_offset
            elif confidence >= self.keep_threshold and jump > self.max_jump:
                # Large jump detected - move gradually toward it
                direction = 1 if new_offset > self._current_offset else -1
                self._current_offset += direction * self.max_jump
                if self.verbose:
                    print(f"  [OffsetTracker] frame={frame_num:6d}  large jump detected "
                          f"({jump} frames), moving gradually to {self._current_offset:+d}")

            self._anchors.append(AnchorPoint(
                frame_num=frame_num, offset=self._current_offset,
                confidence=confidence, searched=True,
            ))
            if self.verbose:
                status = "DEAD" if self.is_exhausted else f"ncc={confidence:.3f}"
                print(f"  [OffsetTracker] frame={frame_num:6d}  offset={self._current_offset:+4d}  {status}")

        return self._current_offset

    def save_log(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_num", "offset", "confidence", "searched"])
            writer.writeheader()
            for a in self._anchors:
                writer.writerow({
                    "frame_num":  a.frame_num,
                    "offset":     a.offset,
                    "confidence": f"{a.confidence:.4f}",
                    "searched":   a.searched,
                })
        print(f"Offset log saved → {path}")

    @property
    def anchors(self) -> list[AnchorPoint]:
        return self._anchors

    def _search_best_offset(self, scan_frame_num: int) -> tuple[int, float]:
        """
        Search for the best offset by comparing scan frame to candidate restored frames.

        Uses batch reading for speed: reads all candidate frames in one sequential pass.

        Returns:
            (best_offset, confidence) where confidence is the NCC score
        """
        # Read the scan frame
        scan_frame = _seek_and_read(self._scan_cap, scan_frame_num)
        if scan_frame is None:
            return self._current_offset, -1.0

        scan_thumb = _to_small_gray(scan_frame, self.resize)

        # Calculate the range of restored frames to check
        # restored_frame = scan_frame + offset
        center_restored = scan_frame_num + self._current_offset
        start_restored = center_restored - self.search_range
        end_restored = center_restored + self.search_range
        num_candidates = end_restored - start_restored + 1

        # Batch read all candidate frames (fast sequential read)
        candidates = _batch_read_frames(self._restored_cap, start_restored, num_candidates)

        # Score each candidate
        best_offset = self._current_offset
        best_score = -1.0

        for restored_idx, restored_frame in candidates:
            if restored_frame is None:
                continue

            restored_thumb = _to_small_gray(restored_frame, self.resize)
            score = _ncc(scan_thumb, restored_thumb)

            if score > best_score:
                best_score = score
                # offset = restored_frame_idx - scan_frame_num
                best_offset = restored_idx - scan_frame_num

        return best_offset, max(0.0, best_score)


def plot_offset_log(csv_path: str, output_path: str = None):
    import matplotlib.pyplot as plt

    frame_nums, offsets, confidences = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["searched"] == "True":
                frame_nums.append(int(row["frame_num"]))
                offsets.append(int(row["offset"]))
                confidences.append(float(row["confidence"]))

    if not frame_nums:
        print("No searched anchor points found in log.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Temporal Offset Drift", fontsize=14, fontweight="bold")

    ax1.plot(frame_nums, offsets, "b-o", markersize=4, linewidth=1.5)
    ax1.set_ylabel("Offset (frames)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(np.mean(offsets), color="r", linestyle="--", linewidth=1,
                label=f"mean = {np.mean(offsets):.1f}")
    ax1.legend(fontsize=9)

    ax2.plot(frame_nums, confidences, "g-o", markersize=4, linewidth=1.5)
    ax2.set_ylabel("NCC confidence")
    ax2.set_xlabel("Scan frame number")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.35, color="orange", linestyle="--", linewidth=1, label="low-conf threshold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {output_path}")
    else:
        plt.show()
    plt.close()
