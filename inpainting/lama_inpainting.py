"""
LaMa (Large Mask Inpainting) for Film Restoration
Fast CNN-based inpainting that works well on high-resolution images and complex textures.
Requires the simple-lama-inpainting package or lama-cleaner.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LamaInpainter:
    """
    LaMa-based inpainting wrapper.

    Note: This requires installing one of:
    1. simple-lama-inpainting: pip install simple-lama-inpainting
    2. lama-cleaner: pip install lama-cleaner

    LaMa is resolution-flexible and works well up to 2K images.
    """

    def __init__(
        self,
        device: str = "cuda",
        backend: str = "simple-lama"  # "simple-lama" or "lama-cleaner"
    ):
        """
        Initialize LaMa inpainter.

        Args:
            device: Device to run on (cuda/cpu)
            backend: Which LaMa implementation to use
        """
        self.device = device
        self.backend = backend
        self.model = None

        if backend == "simple-lama":
            self._init_simple_lama()
        elif backend == "lama-cleaner":
            self._init_lama_cleaner()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_simple_lama(self):
        """Initialize simple-lama-inpainting backend"""
        try:
            from simple_lama_inpainting import SimpleLama
            logger.info("Loading SimpleLama model")
            self.model = SimpleLama()
            logger.info("SimpleLama loaded successfully")
        except ImportError:
            logger.error("simple-lama-inpainting not installed")
            logger.error("Install with: pip install simple-lama-inpainting")
            raise

    def _init_lama_cleaner(self):
        """Initialize lama-cleaner backend"""
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config

            logger.info("Loading LaMa via lama-cleaner")
            self.model = ModelManager(
                name="lama",
                device=self.device
            )
            self.lama_config = Config(
                ldm_steps=25,
                ldm_sampler="plms",
                hd_strategy="Original",
                hd_strategy_crop_margin=128,
                hd_strategy_crop_trigger_size=800,
                hd_strategy_resize_limit=2048
            )
            logger.info("LaMa-cleaner loaded successfully")
        except ImportError:
            logger.error("lama-cleaner not installed")
            logger.error("Install with: pip install lama-cleaner")
            raise

    def inpaint(
        self,
        scan_image: Union[str, Path, Image.Image, np.ndarray],
        mask: Union[str, Path, Image.Image, np.ndarray],
        restored_reference: Optional[Union[str, Path, Image.Image, np.ndarray]] = None
    ) -> Image.Image:
        """
        Inpaint defects in scan using LaMa.

        Args:
            scan_image: Input scan with defects (the image to restore)
            mask: Binary mask (white=defect to fill, black=preserve)
            restored_reference: Optional restored/BluRay reference (note: LaMa doesn't use this directly,
                               but included for API consistency)

        Returns:
            Inpainted PIL Image (restored scan)
        """
        # Load and convert scan
        if not isinstance(scan_image, Image.Image):
            if isinstance(scan_image, (str, Path)):
                scan_image = Image.open(scan_image).convert("RGB")
            elif isinstance(scan_image, np.ndarray):
                scan_image = Image.fromarray(scan_image).convert("RGB")

        # Load and convert mask
        if not isinstance(mask, Image.Image):
            if isinstance(mask, (str, Path)):
                mask = Image.open(mask).convert("L")
            elif isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask).convert("L")

        # Ensure sizes match
        if scan_image.size != mask.size:
            logger.warning(f"Resizing mask from {mask.size} to {scan_image.size}")
            mask = mask.resize(scan_image.size, Image.LANCZOS)

        if restored_reference is not None:
            logger.info("Note: LaMa doesn't use restored reference directly, inpainting scan only")

        logger.info(f"Inpainting scan with LaMa, image size: {scan_image.size}")

        # Convert to numpy for processing
        scan_np = np.array(scan_image)
        mask_np = np.array(mask)

        # Run inpainting based on backend
        if self.backend == "simple-lama":
            result_np = self._inpaint_simple_lama(scan_np, mask_np)
        elif self.backend == "lama-cleaner":
            result_np = self._inpaint_lama_cleaner(scan_np, mask_np)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return Image.fromarray(result_np)

    def _inpaint_simple_lama(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Inpaint using simple-lama-inpainting"""
        # SimpleLama expects mask as single channel
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Run inpainting
        result = self.model(image, mask)
        return result

    def _inpaint_lama_cleaner(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Inpaint using lama-cleaner"""
        # Ensure mask is binary (0 or 255)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.uint8) * 255

        # Run inpainting
        result = self.model(image, mask, self.lama_config)
        return result

    def inpaint_batch(
        self,
        scan_paths: list,
        mask_paths: list,
        output_dir: Union[str, Path],
        restored_reference_paths: Optional[list] = None
    ):
        """
        Batch inpainting.

        Args:
            scan_paths: Input scan paths (to restore)
            mask_paths: Corresponding mask paths
            output_dir: Output directory
            restored_reference_paths: Optional restored references (not used by LaMa)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        assert len(scan_paths) == len(mask_paths), "Scan and mask count mismatch"

        for i, (scan_path, mask_path) in enumerate(zip(scan_paths, mask_paths)):
            logger.info(f"Processing {i+1}/{len(scan_paths)}: {scan_path}")

            result = self.inpaint(scan_path, mask_path)

            img_name = Path(scan_path).name
            output_path = output_dir / img_name
            result.save(output_path)
            logger.info(f"Saved to {output_path}")


class LamaInpainterFallback:
    """
    Fallback implementation that provides interface compatibility
    when LaMa packages are not installed. Raises helpful error messages.
    """

    def __init__(self, *args, **kwargs):
        logger.warning("LaMa packages not available - using fallback")
        logger.warning("Install with: pip install simple-lama-inpainting")
        logger.warning("Or: pip install lama-cleaner")

    def inpaint(self, image, mask):
        raise RuntimeError(
            "LaMa inpainting requires installing:\n"
            "  pip install simple-lama-inpainting\n"
            "or:\n"
            "  pip install lama-cleaner\n"
            "Please install one of these packages to use LaMa inpainting."
        )

    def inpaint_batch(self, *args, **kwargs):
        self.inpaint(None, None)


def main():
    """Example usage"""
    try:
        inpainter = LamaInpainter(device="cuda", backend="simple-lama")
    except ImportError:
        print("LaMa not available. Install with:")
        print("  pip install simple-lama-inpainting")
        return

    # Example: inpaint scan defects
    scan = "path/to/scan_with_defects.png"
    defect_mask = "path/to/defect_mask.png"

    result = inpainter.inpaint(
        scan_image=scan,
        mask=defect_mask
    )

    result.save("lama_inpainted_scan.png")
    print("LaMa scan inpainting complete!")


if __name__ == "__main__":
    main()
