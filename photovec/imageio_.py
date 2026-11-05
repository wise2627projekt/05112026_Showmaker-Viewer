"""Image loading (JPG/PNG/HEIC/TIFF, EXIF orientation honoured)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
try:  # optional HEIC support
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass


def load_image_alpha(path: str, target: Optional[Tuple[int, int]] = None):
    """Return (rgb uint8, alpha uint8 or None).  RGB is *straight* (un-premultiplied) colour;
    alpha is None when the image is fully opaque.  EXIF orientation is applied.
    `target` = (w, h) the picture will be shown at: big photos are shrunk while loading (JPEG draft mode + box reduce,
    at least 2x the needed size) so a 40-megapixel photo never sits in memory at full size."""
    try:
        with Image.open(path) as im:
            if target is not None:
                try:
                    im.draft("RGB", (target[0] * 2, target[1] * 2))
                except Exception:
                    pass
            im = ImageOps.exif_transpose(im)
            if target is not None:
                f = int(min(im.width / (2.0 * target[0]), im.height / (2.0 * target[1])))
                if f >= 2:
                    im = im.reduce(f)
            has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            arr = np.asarray(im.convert("RGBA" if has_alpha else "RGB"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot read image '{path}': {exc}") from exc
    if arr.shape[2] == 4:
        a = arr[..., 3]
        return np.ascontiguousarray(arr[..., :3]), (None if a.min() == 255 else a.copy())
    return arr, None


def load_image(path: str) -> np.ndarray:
    """RGB uint8 with EXIF orientation applied; transparency is composited onto white."""
    rgb, a = load_image_alpha(path)
    if a is None:
        return rgb
    k = (a.astype(np.float32) / 255.0)[..., None]
    return np.rint(rgb * k + 255.0 * (1.0 - k)).astype(np.uint8)
