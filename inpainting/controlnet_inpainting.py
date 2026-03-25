"""
ControlNet-guided Inpainting for Film Restoration
Uses ControlNet to better preserve reference image structure during inpainting.
"""

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ControlNetInpainter:
    """
    ControlNet-guided inpainting that uses the aligned BluRay/restored frame
    as a control signal to better preserve structure and details.
    """

    def __init__(
        self,
        controlnet_model_id: str = "lllyasviel/control_v11p_sd15_inpaint",
        base_model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str = "cuda",
        torch_dtype = torch.float16
    ):
        """
        Initialize ControlNet inpainting pipeline.

        Args:
            controlnet_model_id: ControlNet model for inpainting
            base_model_id: Base SD inpainting model
            device: Device to run on
            torch_dtype: Data type for inference
        """
        logger.info(f"Loading ControlNet: {controlnet_model_id}")
        self.device = device

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=torch_dtype
        )

        # Load pipeline with ControlNet
        logger.info(f"Loading base model: {base_model_id}")
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype
        ).to(device)

        # Enable optimizations
        if device == "cuda":
            logger.info("Enabling memory optimizations")
            self.pipe.enable_attention_slicing()
            # Uncomment if xformers installed
            # self.pipe.enable_xformers_memory_efficient_attention()

        logger.info("Pipeline loaded successfully")

    @staticmethod
    def make_inpaint_condition(
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        Create control image for ControlNet inpainting.
        Marks defect regions with -1 (gray) as conditioning signal.

        Args:
            image: Original image (RGB)
            mask: Binary mask (white=defect, black=preserve)

        Returns:
            Control image with masked regions marked
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image).astype(np.float32)
        else:
            image_np = image.astype(np.float32)

        if isinstance(mask, Image.Image):
            mask_np = np.array(mask).astype(np.float32) / 255.0
        else:
            mask_np = mask.astype(np.float32) / 255.0

        # Expand mask if grayscale
        if mask_np.ndim == 2:
            mask_np = mask_np[:, :, np.newaxis]

        # Set masked regions to gray (127.5)
        control = image_np.copy()
        control[mask_np[:, :, 0] > 0.5] = 127.5

        control = np.clip(control, 0, 255).astype(np.uint8)
        return Image.fromarray(control)

    def inpaint(
        self,
        scan_image: Union[str, Path, Image.Image, np.ndarray],
        mask: Union[str, Path, Image.Image, np.ndarray],
        restored_reference: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "film still without defects, clean photorealistic 35mm film frame",
        negative_prompt: str = "ugly, blurred, cartoon, low quality, artifacts",
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        strength: float = 0.99,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Inpaint defects in scan using ControlNet with restored reference guidance.

        Args:
            scan_image: Input scan with defects (the image to restore)
            mask: Binary defect mask (white=defect, black=preserve)
            restored_reference: BluRay/restored frame used as ControlNet guidance
            prompt: Text prompt
            negative_prompt: Negative prompt
            guidance_scale: Prompt adherence strength
            controlnet_conditioning_scale: ControlNet influence (0-1)
            strength: Inpainting strength
            num_inference_steps: Diffusion steps
            seed: Random seed

        Returns:
            Inpainted scan (restored)
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

        # Load and convert restored reference
        if not isinstance(restored_reference, Image.Image):
            if isinstance(restored_reference, (str, Path)):
                restored_reference = Image.open(restored_reference).convert("RGB")
            elif isinstance(restored_reference, np.ndarray):
                restored_reference = Image.fromarray(restored_reference).convert("RGB")

        # Ensure sizes match
        if scan_image.size != mask.size:
            logger.warning(f"Resizing mask from {mask.size} to {scan_image.size}")
            mask = mask.resize(scan_image.size, Image.LANCZOS)

        if restored_reference.size != scan_image.size:
            logger.warning(f"Resizing restored reference from {restored_reference.size} to {scan_image.size}")
            restored_reference = restored_reference.resize(scan_image.size, Image.LANCZOS)

        # Create control image from restored reference
        # The control image shows what the clean frame should look like
        logger.info("Creating control image from restored reference and mask")
        control_image = self.make_inpaint_condition(restored_reference, mask)

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            logger.info(f"Using seed: {seed}")

        logger.info(f"Inpainting scan of size {scan_image.size}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"ControlNet scale: {controlnet_conditioning_scale}")

        # Run ControlNet inpainting on the scan, guided by the restored reference
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=scan_image,
            mask_image=mask,
            control_image=control_image,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

        return result

    def inpaint_batch(
        self,
        scan_paths: list,
        mask_paths: list,
        restored_reference_paths: list,
        output_dir: Union[str, Path],
        **kwargs
    ):
        """
        Batch inpainting with ControlNet.

        Args:
            scan_paths: Input scan paths (to restore)
            mask_paths: Mask paths
            restored_reference_paths: Restored/BluRay frame paths for guidance
            output_dir: Output directory
            **kwargs: Additional arguments for inpaint()
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        assert len(scan_paths) == len(mask_paths), "Scan and mask count mismatch"
        assert len(scan_paths) == len(restored_reference_paths), "Scan and reference count mismatch"

        for i, (scan_path, mask_path, ref_path) in enumerate(zip(scan_paths, mask_paths, restored_reference_paths)):
            logger.info(f"Processing {i+1}/{len(scan_paths)}: {scan_path}")

            result = self.inpaint(scan_path, mask_path, ref_path, **kwargs)

            img_name = Path(scan_path).name
            output_path = output_dir / img_name
            result.save(output_path)
            logger.info(f"Saved to {output_path}")


def main():
    """Example usage"""
    inpainter = ControlNetInpainter(device="cuda")

    # Example: inpaint scan defects guided by restored BluRay frame
    scan = "path/to/scan_with_defects.png"
    defect_mask = "path/to/defect_mask.png"
    restored_bluray = "path/to/restored_bluray_frame.png"

    result = inpainter.inpaint(
        scan_image=scan,
        mask=defect_mask,
        restored_reference=restored_bluray,
        prompt="film still without defects, clean photorealistic 35mm film frame",
        controlnet_conditioning_scale=0.8,
        guidance_scale=7.5,
        seed=42
    )

    result.save("controlnet_inpainted_scan.png")
    print("ControlNet scan inpainting complete!")


if __name__ == "__main__":
    main()
