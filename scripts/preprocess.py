"""
Step 1: Morphological Preprocessing for Epigrascan
Takes raw estampages and high-resolution rock photos and applies the "Morphological Sandwich"
to output high-contrast, binarized text for the CNN.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def morphological_sandwich(img_path):
    """
    Applies lighting correction, pre-blurring, and heavy binarization to extract
    clean Brahmi strokes from raw stone photographs or paper rubbings.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read {img_path}")

    # Standardize Resolution (Scale down massive phone photos)
    max_dimension = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # 1. Lightness Adjustment via CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # 2. Pre-Threshold Blurring (Drops granular rock noise)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Massive Adaptive Binarization
    # Block Size 75 captures thick carved strokes. C=15 drops rock texture.
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        75,
        15
    )

    # 4. Morphological Dilation & Final Blur (Bridges micro-fractures)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    cleaned = cv2.medianBlur(dilated, 3)

    return cleaned

def main():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_estampages"])
    proc_dir = Path(config["paths"]["processed"])

    proc_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(raw_dir.glob("*.[jp][pn]*"))
    print(f"Found {len(image_files)} raw estampage images to process...\n")

    for img_path in tqdm(image_files, desc="Preprocessing estampages"):
        try:
            # We pass ONLY the path (1 argument)
            processed_img = morphological_sandwich(img_path)

            # Save the pure binary image
            out_path = proc_dir / f"{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), processed_img)

        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")

    print(f"\n✅ Preprocessing completed! Cleaned images saved in: {proc_dir}")

if __name__ == "__main__":
    main()