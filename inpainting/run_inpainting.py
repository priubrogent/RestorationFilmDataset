"""
CLI entry point for film defect inpainting.

Supports three backends:
  - lama       Fast CNN, no reference needed, ~1-3s/frame, ~2-4 GB VRAM
  - sdxl       Best quality diffusion, ~10-15s/frame, ~8-10 GB VRAM
  - controlnet Uses restored frame as structural guide, ~8-12s/frame, ~6-8 GB VRAM

Usage examples
--------------
# Single frame with LaMa (fastest):
python run_inpainting.py --method lama \\
    --scan   ../../TemporalAligment/batch_masks_f450_b400_ecc/scan_000450.png \\
    --mask   ../../TemporalAligment/batch_masks_f450_b400_ecc/mask_000450.png \\
    --output ./results

# Single frame with SDXL (best quality):
python run_inpainting.py --method sdxl \\
    --scan      path/to/scan.png \\
    --mask      path/to/mask.png \\
    --restored  path/to/restored_reference.png \\
    --output    ./results

# Single frame with ControlNet (best structure preservation):
python run_inpainting.py --method controlnet \\
    --scan      path/to/scan.png \\
    --mask      path/to/mask.png \\
    --restored  path/to/restored_reference.png \\
    --output    ./results

# Compare all three methods on the same frame:
python run_inpainting.py --method all \\
    --scan     path/to/scan.png \\
    --mask     path/to/mask.png \\
    --restored path/to/restored_reference.png \\
    --output   ./results

# Batch with LaMa:
python run_inpainting.py --method lama --batch \\
    --scan-dir  ./masks_output \\
    --mask-dir  ./masks_output \\
    --scan-glob "scan_*.png" \\
    --mask-glob "mask_*.png" \\
    --output    ./inpainted
"""

import argparse
import glob
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-image helpers
# ---------------------------------------------------------------------------

def run_lama(scan_path, mask_path, output_path, backend="simple-lama"):
    from lama_inpainting import LamaInpainter
    t0 = time.time()
    inpainter = LamaInpainter(device="cuda", backend=backend)
    result = inpainter.inpaint(scan_image=scan_path, mask=mask_path)
    result.save(output_path)
    logger.info(f"LaMa done in {time.time()-t0:.1f}s → {output_path}")
    return result


def run_sdxl(scan_path, mask_path, output_path, restored_path=None,
             prompt="realistic 35mm film frame, clean photorealistic image, no scratches or dust",
             negative_prompt="ugly, blurred, cartoon, artificial, low quality, compression artifacts",
             guidance_scale=8.0, strength=0.99, num_steps=50, seed=42):
    from sdxl_inpainting import SDXLInpainter
    t0 = time.time()
    inpainter = SDXLInpainter(device="cuda")
    result = inpainter.inpaint(
        scan_image=scan_path,
        mask=mask_path,
        restored_reference=restored_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=num_steps,
        preserve_unmasked=True,
        seed=seed,
    )
    result.save(output_path)
    logger.info(f"SDXL done in {time.time()-t0:.1f}s → {output_path}")
    return result


def run_controlnet(scan_path, mask_path, output_path, restored_path,
                   prompt="film still without defects, clean photorealistic 35mm film frame",
                   negative_prompt="ugly, blurred, cartoon, low quality, artifacts",
                   guidance_scale=7.5, controlnet_scale=0.8, strength=0.99, num_steps=50, seed=42):
    from controlnet_inpainting import ControlNetInpainter
    t0 = time.time()
    inpainter = ControlNetInpainter(device="cuda")
    result = inpainter.inpaint(
        scan_image=scan_path,
        mask=mask_path,
        restored_reference=restored_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        strength=strength,
        num_inference_steps=num_steps,
        seed=seed,
    )
    result.save(output_path)
    logger.info(f"ControlNet done in {time.time()-t0:.1f}s → {output_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inpaint film defects using LaMa, SDXL, or ControlNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", choices=["lama", "sdxl", "controlnet", "all"],
                        default="lama", help="Inpainting method (default: lama)")

    # Single-image mode
    single = parser.add_argument_group("Single image")
    single.add_argument("--scan", help="Path to scan frame with defects")
    single.add_argument("--mask", help="Path to defect mask (white=defect)")
    single.add_argument("--restored", help="Path to restored/BluRay reference (required for sdxl/controlnet)")

    # Batch mode
    batch = parser.add_argument_group("Batch mode")
    batch.add_argument("--batch", action="store_true", help="Enable batch processing")
    batch.add_argument("--scan-dir", help="Directory containing scan frames")
    batch.add_argument("--mask-dir", help="Directory containing mask files")
    batch.add_argument("--restored-dir", help="Directory containing restored reference frames")
    batch.add_argument("--scan-glob", default="scan_*.png", help="Glob pattern for scans")
    batch.add_argument("--mask-glob", default="mask_*.png", help="Glob pattern for masks")
    batch.add_argument("--restored-glob", default="restored_*.png", help="Glob pattern for restored refs")

    parser.add_argument("--output", default="./inpainting_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for diffusion methods")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- Batch mode ----
    if args.batch:
        if not args.scan_dir or not args.mask_dir:
            parser.error("--batch requires --scan-dir and --mask-dir")

        scan_files = sorted(glob.glob(os.path.join(args.scan_dir, args.scan_glob)))
        mask_files = sorted(glob.glob(os.path.join(args.mask_dir, args.mask_glob)))

        if len(scan_files) != len(mask_files):
            parser.error(f"Scan count ({len(scan_files)}) != mask count ({len(mask_files)})")

        restored_files = None
        if args.restored_dir:
            restored_files = sorted(glob.glob(os.path.join(args.restored_dir, args.restored_glob)))
            if len(restored_files) != len(scan_files):
                parser.error(f"Restored count ({len(restored_files)}) != scan count ({len(scan_files)})")

        logger.info(f"Batch: {len(scan_files)} frames with method={args.method}")
        method = args.method if args.method != "all" else "lama"

        for i, (scan_p, mask_p) in enumerate(zip(scan_files, mask_files)):
            stem = Path(scan_p).stem.replace("scan_", "")
            out_p = os.path.join(args.output, f"{method}_{stem}.png")
            ref_p = restored_files[i] if restored_files else None

            if method == "lama":
                run_lama(scan_p, mask_p, out_p)
            elif method == "sdxl":
                run_sdxl(scan_p, mask_p, out_p, restored_path=ref_p, seed=args.seed)
            elif method == "controlnet":
                if ref_p is None:
                    logger.error("ControlNet requires --restored-dir")
                    break
                run_controlnet(scan_p, mask_p, out_p, ref_p, seed=args.seed)

        logger.info(f"Batch complete → {args.output}")
        return

    # ---- Single image mode ----
    if not args.scan or not args.mask:
        parser.error("Single-image mode requires --scan and --mask")

    methods = ["sdxl", "controlnet", "lama"] if args.method == "all" else [args.method]

    for method in methods:
        out_p = os.path.join(args.output, f"{method}_result.png")
        try:
            if method == "lama":
                run_lama(args.scan, args.mask, out_p)
            elif method == "sdxl":
                run_sdxl(args.scan, args.mask, out_p, restored_path=args.restored, seed=args.seed)
            elif method == "controlnet":
                if not args.restored:
                    logger.error("ControlNet requires --restored")
                    continue
                run_controlnet(args.scan, args.mask, out_p, args.restored, seed=args.seed)
        except Exception as e:
            logger.error(f"{method} failed: {e}")


if __name__ == "__main__":
    main()
