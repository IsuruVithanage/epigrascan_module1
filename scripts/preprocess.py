"""
Step 2: Morphological Sandwich Preprocessor for Epigrascan Vision Module
Cleans raw estampage images: CLAHE → Adaptive Binarization → Dilation → Median Blur
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def morphological_sandwich(image_path: str, config: dict):
    """Apply the full Morphological Sandwich pipeline"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 1. CLAHE - Fix uneven lighting on stone rubbings
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Adaptive Gaussian Thresholding (text = white, background = black)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 3. Dilation - Thicken faint or broken strokes
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # 4. Median Blur - Remove paper artifacts and salt-pepper noise
    cleaned = cv2.medianBlur(dilated, 3)

    return cleaned


def main():
    config = load_config()

    # Create output folder if it doesn't exist
    processed_dir = Path(config["paths"]["processed"])
    processed_dir.mkdir(exist_ok=True)

    # Get all raw estampage images
    raw_dir = Path(config["paths"]["raw_estampages"])
    image_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg"))

    print(f"Found {len(image_files)} raw estampage images to process...\n")

    for img_path in tqdm(image_files, desc="Preprocessing estampages"):
        try:
            cleaned = morphological_sandwich(str(img_path), config)

            # Save cleaned image
            output_path = processed_dir / img_path.name
            cv2.imwrite(str(output_path), cleaned)

        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    print(f"\n✅ Preprocessing completed! Cleaned images saved in: {processed_dir}")


if __name__ == "__main__":
    main()