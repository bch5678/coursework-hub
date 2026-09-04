"""Deterministic grayscale decoding; no population-derived statistics."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def decode_image(path: str | Path) -> Image.Image:
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        import pydicom
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
        ds = pydicom.dcmread(path)
        if getattr(ds, "PhotometricInterpretation", "") not in {"MONOCHROME1", "MONOCHROME2"}:
            raise ValueError("Only monochrome DICOM chest radiographs are supported")
        raw = ds.pixel_array
        if raw.ndim != 2:
            raise ValueError("Only single-frame 2D DICOM images are supported")
        padding = np.zeros(raw.shape, dtype=bool)
        if hasattr(ds, "PixelPaddingValue"):
            lower = ds.PixelPaddingValue
            upper = getattr(ds, "PixelPaddingRangeLimit", lower)
            padding = (raw >= min(lower, upper)) & (raw <= max(lower, upper))
        pixels = np.asarray(apply_voi_lut(apply_modality_lut(raw, ds), ds), dtype=np.float32)
        usable = ~padding & np.isfinite(pixels)
        if not usable.any() or not np.isfinite(pixels[~padding]).all():
            raise ValueError("No finite DICOM pixels")
        lo, hi = float(pixels[usable].min()), float(pixels[usable].max())
        if hi <= lo:
            raise ValueError("Constant DICOM image")
        pixels = np.clip((pixels - lo) / (hi - lo), 0, 1)
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixels = 1 - pixels
        pixels[padding] = 0
        result = Image.fromarray(np.round(pixels * 255).astype(np.uint8))
    else:
        with Image.open(path) as image:
            image.load()
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images are unsupported")
            result = image.convert("L")
    if min(result.size) < 2 or result.getextrema()[0] == result.getextrema()[1]:
        raise ValueError("Empty, too small or constant image")
    return result


def image_array(path, image_size: int, channels: int) -> np.ndarray:
    image = ImageOps.pad(decode_image(path), (image_size, image_size), method=Image.Resampling.BILINEAR, color=0)
    array = np.asarray(image, dtype=np.float32)[None, :, :] / 255.0
    return np.repeat(array, channels, axis=0) if channels == 3 else array.copy()
