"""
Stable Diffusion XL Inpainting Pipeline for Film Restoration
Uses diffusers library to inpaint defects detected in film scans.
"""

import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDXLInpainter:
    """
    Stable Diffusion XL-based inpainting for film restoration.
    Fills masked defect regions while preserving the rest of the frame.
    """

    def __init__(
        self,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device: str = "cuda",
        torch_dtype = torch.float16
    ):
        """
        Initialize the SDXL inpainting pipeline.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run on (cuda/cpu)
            torch_dtype: Torch data type for inference
        """
        logger.info(f"Loading SDXL inpainting model: {model_id}")
        self.device = device
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None
        ).to(device)

        # Enable optimizations
        if device == "cuda":
            logger.info("Enabling memory optimizations")
            self.pipe.enable_attention_slicing()
            # Uncomment if you have xformers installed
            # self.pipe.enable_xformers_memory_efficient_attention()

        logger.info("Pipeline loaded successfully")

    def inpaint(
        self,
        scan_image: Union[str, Path, Image.Image, np.ndarray],
        mask: Union[str, Path, Image.Image, np.ndarray],
        restored_reference: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        prompt: str = "realistic 35mm film frame, clean photorealistic image, no scratches or dust",
        negative_prompt: str = "ugly, blurred, cartoon, artificial, low quality, compression artifacts",
        guidance_scale: float = 8.0,
        strength: float = 0.99,
        num_inference_steps: int = 50,
        preserve_unmasked: bool = True,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Inpaint defects in the scan using the mask, optionally guided by a restored reference.

        Args:
            scan_image: Input scan with defects (the image to restore)
            mask: Binary mask (white=defect to fill, black=preserve)
            restored_reference: Optional BluRay/restored frame to guide inpainting (uses as init if provided)
            prompt: Text prompt describing desired output
            negative_prompt: What to avoid in generation
            guidance_scale: How strongly to follow the prompt (7-15 typical)
            strength: How much to alter masked regions (0.99 recommended)
            num_inference_steps: Diffusion steps (more = better quality, slower)
            preserve_unmasked: Apply overlay to ensure unmasked areas unchanged
            seed: Random seed for reproducibility

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
            logger.warning(f"Image size {scan_image.size} != mask size {mask.size}, resizing mask")
            mask = mask.resize(scan_image.size, Image.LANCZOS)

        # If restored reference provided, use it as initialization
        # Otherwise, use the scan itself
        init_image = scan_image
        if restored_reference is not None:
            if not isinstance(restored_reference, Image.Image):
                if isinstance(restored_reference, (str, Path)):
                    restored_reference = Image.open(restored_reference).convert("RGB")
                elif isinstance(restored_reference, np.ndarray):
                    restored_reference = Image.fromarray(restored_reference).convert("RGB")

            if restored_reference.size != scan_image.size:
                logger.warning(f"Resizing restored reference from {restored_reference.size} to {scan_image.size}")
                restored_reference = restored_reference.resize(scan_image.size, Image.LANCZOS)

            # Use restored reference as initialization
            init_image = restored_reference
            logger.info("Using restored reference frame as initialization")

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            logger.info(f"Using seed: {seed}")

        logger.info(f"Inpainting scan of size {scan_image.size}")
        logger.info(f"Prompt: {prompt}")

        # Run inpainting on the initialization image (scan or restored reference)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

        # Apply overlay to preserve unmasked regions from the SCAN
        if preserve_unmasked:
            logger.info("Applying overlay to preserve unmasked regions from scan")
            result = self.apply_overlay(scan_image, result, mask)

        return result

    @staticmethod
    def apply_overlay(
        original: Image.Image,
        generated: Image.Image,
        mask: Image.Image
    ) -> Image.Image:
        """
        Overlay original image onto generated, using mask.
        Only masked (white) regions from generated are kept.

        Args:
            original: Original input image
            generated: Generated inpainted image
            mask: Binary mask (white=use generated, black=use original)

        Returns:
            Composited image
        """
        # Convert to numpy for easy blending
        orig_np = np.array(original, dtype=np.float32)
        gen_np = np.array(generated, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        # Expand mask to RGB if needed
        if mask_np.ndim == 2:
            mask_np = mask_np[:, :, np.newaxis]

        # Blend: use generated where mask is white, original elsewhere
        result_np = mask_np * gen_np + (1 - mask_np) * orig_np
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)

        return Image.fromarray(result_np)

    def inpaint_batch(
        self,
        scan_paths: list,
        mask_paths: list,
        output_dir: Union[str, Path],
        restored_reference_paths: Optional[list] = None,
        **kwargs
    ):
        """
        Inpaint a batch of scans.

        Args:
            scan_paths: List of scan image paths (to restore)
            mask_paths: List of corresponding mask paths
            output_dir: Directory to save results
            restored_reference_paths: Optional list of restored/BluRay frames for guidance
            **kwargs: Additional arguments passed to inpaint()
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        assert len(scan_paths) == len(mask_paths), "Must have same number of scans and masks"

        if restored_reference_paths is not None:
            assert len(scan_paths) == len(restored_reference_paths), "Must have same number of scans and references"

        for i, (scan_path, mask_path) in enumerate(zip(scan_paths, mask_paths)):
            logger.info(f"Processing {i+1}/{len(scan_paths)}: {scan_path}")

            restored_ref = restored_reference_paths[i] if restored_reference_paths else None
            result = self.inpaint(scan_path, mask_path, restored_reference=restored_ref, **kwargs)

            # Save with same name as input
            img_name = Path(scan_path).name
            output_path = output_dir / img_name
            result.save(output_path)
            logger.info(f"Saved to {output_path}")


def main():
    """Example usage"""
    # Initialize inpainter
    inpainter = SDXLInpainter(device="cuda")

    # Example: inpaint defects in scan, guided by restored BluRay frame
    scan_path = "path/to/scan_with_defects.png"
    defect_mask_path = "path/to/defect_mask.png"
    restored_bluray_path = "path/to/restored_bluray_frame.png"

    result = inpainter.inpaint(
        scan_image=scan_path,
        mask=defect_mask_path,
        restored_reference=restored_bluray_path,
        prompt="realistic 35mm film frame, clean photorealistic image, no scratches or dust",
        guidance_scale=8.0,
        strength=0.99,
        num_inference_steps=50,
        seed=42
    )

    result.save("inpainted_scan_result.png")
    print("Scan inpainting complete!")


if __name__ == "__main__":
    main()
