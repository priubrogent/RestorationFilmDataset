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

Strictness improvements
------------------------
- Cached VideoCapture: caps opened once in __init__, never closed until __del__.
  Eliminates the dominant per-frame overhead and allows more frames per anchor.
- Multi-frame NCC: for each candidate offset, NCC is averaged across
  `2*n_verify + 1` scan frames (anchor ± n_verify × verify_step). Averaging
  suppresses noise from motion blur or bad frames at a single point.
- Larger default search_range (20) and denser anchor_interval (25).
- Higher-resolution thumbnails (240×135 instead of 160×90).
- min_conf retry: if best NCC after full search < min_conf, the range doubles
  and the search is repeated once before accepting the result.
- keep_threshold: if the final best NCC is still below keep_threshold the
  current offset is NOT updated (previous value preserved).
- is_exhausted: True when scan returns None for 3+ consecutive anchor searches.
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


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    if frame_idx < 0:
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


@dataclass
class AnchorPoint:
    frame_num:  int
    offset:     int
    confidence: float
    searched:   bool


class TemporalOffsetTracker:
    def __init__(
        self,
        scan_path: str,
        restored_path: str,
        initial_offset: int,
        anchor_interval: int = 25,
        search_range: int = 20,
        resize: tuple = (240, 135),
        verbose: bool = True,
        low_conf_threshold: float = 0.35,
        low_conf_patience: int = 2,
        max_search_multiplier: int = 4,
        n_verify: int = 2,
        verify_step: int = 4,
        min_conf: float = 0.30,
        keep_threshold: float = 0.20,
    ):
        self.scan_path       = scan_path
        self.restored_path   = restored_path
        self.anchor_interval = anchor_interval
        self.search_range    = search_range
        self.resize          = resize
        self.verbose         = verbose
        self.low_conf_threshold   = low_conf_threshold
        self.low_conf_patience    = low_conf_patience
        self.max_search_multiplier = max_search_multiplier
        self.n_verify        = n_verify
        self.verify_step     = verify_step
        self.min_conf        = min_conf
        self.keep_threshold  = keep_threshold

        self._current_offset     = initial_offset
        self._range_multiplier   = 1
        self._consecutive_low    = 0
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
        if frame_num % self.anchor_interval == 0:
            new_offset, confidence = self._search_best_offset(frame_num, self._current_offset)

            if confidence == -1.0:
                self._consecutive_none += 1
                confidence = 0.0
            else:
                self._consecutive_none = 0
                if confidence < self.low_conf_threshold:
                    self._consecutive_low += 1
                    if self._consecutive_low >= self.low_conf_patience:
                        self._range_multiplier = min(
                            self._range_multiplier * 2,
                            self.max_search_multiplier,
                        )
                        if self.verbose:
                            eff = self.search_range * self._range_multiplier
                            print(f"  [OffsetTracker] low confidence ({confidence:.3f}) "
                                  f"— expanding search to ±{eff}")
                else:
                    self._consecutive_low = 0
                    self._range_multiplier = 1

                if confidence >= self.keep_threshold:
                    self._current_offset = new_offset
                else:
                    if self.verbose:
                        print(f"  [OffsetTracker] frame={frame_num:6d}  "
                              f"confidence={confidence:.3f} below keep_threshold={self.keep_threshold} "
                              f"— keeping offset={self._current_offset:+d}")

            self._anchors.append(AnchorPoint(
                frame_num=frame_num, offset=new_offset,
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

    def _collect_scan_thumbs(self, anchor_fn: int) -> list[np.ndarray]:
        thumbs = []
        for k in range(-self.n_verify, self.n_verify + 1):
            fn = anchor_fn + k * self.verify_step
            frame = _read_frame(self._scan_cap, fn)
            if frame is not None:
                thumbs.append(_to_small_gray(frame, self.resize))
        return thumbs

    def _score_offset(self, scan_thumbs: list[np.ndarray], anchor_fn: int, test_offset: int) -> float:
        scores = []
        for k in range(-self.n_verify, self.n_verify + 1):
            scan_idx = self.n_verify + k
            if scan_idx >= len(scan_thumbs):
                continue
            fn = anchor_fn + k * self.verify_step
            rest_frame = _read_frame(self._restored_cap, fn + test_offset)
            if rest_frame is None:
                continue
            scores.append(_ncc(scan_thumbs[scan_idx], _to_small_gray(rest_frame, self.resize)))
        return float(np.mean(scores)) if scores else 0.0

    def _search_range_for_offset(
        self,
        scan_thumbs: list[np.ndarray],
        anchor_fn: int,
        current_offset: int,
        eff_range: int,
    ) -> tuple[int, float]:
        best_offset = current_offset
        best_score  = -1.0
        for test_offset in range(current_offset - eff_range, current_offset + eff_range + 1):
            score = self._score_offset(scan_thumbs, anchor_fn, test_offset)
            if score > best_score:
                best_score  = score
                best_offset = test_offset
        return best_offset, max(0.0, best_score)

    def _search_best_offset(self, frame_num: int, current_offset: int) -> tuple[int, float]:
        scan_thumbs = self._collect_scan_thumbs(frame_num)
        if not scan_thumbs:
            return current_offset, -1.0

        eff_range = self.search_range * self._range_multiplier
        best_offset, best_score = self._search_range_for_offset(
            scan_thumbs, frame_num, current_offset, eff_range
        )

        if best_score < self.min_conf:
            retry_range = eff_range * 2
            if self.verbose:
                print(f"  [OffsetTracker] frame={frame_num:6d}  score={best_score:.3f} < min_conf "
                      f"— retrying with range ±{retry_range}")
            retry_offset, retry_score = self._search_range_for_offset(
                scan_thumbs, frame_num, current_offset, retry_range
            )
            if retry_score > best_score:
                best_offset, best_score = retry_offset, retry_score

        return best_offset, best_score


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
