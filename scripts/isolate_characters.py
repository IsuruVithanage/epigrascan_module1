"""
Step 3: Contour-Based Character Isolation (Improved & Robust Version)
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def isolate_characters(image_path: str, config: dict, min_area: int = 100):
    """Extract individual characters using contour detection - robust version"""
    try:
        # Read binary image
        binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if binary is None:
            raise ValueError(f"Could not read image")

        # Ensure it's 2D
        if len(binary.shape) != 2:
            binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_crops = []
        positions = []  # for left-to-right sorting
        target_size = config["model"]["image_size"]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Crop character
            crop = binary[y:y + h, x:x + w]

            if crop.size == 0:
                continue

            # Calculate difference between height and width
            diff = abs(h - w)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

            if h > w:  # Taller than it is wide
                pad_left = diff // 2
                pad_right = diff - pad_left
            elif w > h:  # Wider than it is tall
                pad_top = diff // 2
                pad_bottom = diff - pad_top

            # Pad with black (0) to maintain the background geometry
            square_crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                             cv2.BORDER_CONSTANT, value=0)

            # Safely resize the perfect square
            final_crop = cv2.resize(square_crop, (target_size, target_size))

            char_crops.append(final_crop)
            positions.append(x)  # x-position for sorting

        # THE FIX FROM EARLIER: Safely sort left to right avoiding array truth ambiguity
        if positions:
            sorted_chars = [crop for _, crop in sorted(zip(positions, char_crops), key=lambda item: item[0])]
            return sorted_chars
        else:
            return []

    except Exception as e:
        print(f"  Error processing {Path(image_path).name}: {e}")
        return []

def main():
    config = load_config()

    isolated_dir = Path("data/isolated")
    isolated_dir.mkdir(exist_ok=True)

    processed_dir = Path(config["paths"]["processed"])
    image_files = list(processed_dir.glob("*.jpg")) + \
                  list(processed_dir.glob("*.png")) + \
                  list(processed_dir.glob("*.jpeg"))

    print(f"Found {len(image_files)} cleaned images...\n")

    total_chars = 0
    for img_path in tqdm(image_files, desc="Isolating characters"):
        chars = isolate_characters(str(img_path), config, min_area=100)
        base_name = img_path.stem

        for i, char_img in enumerate(chars):
            output_path = isolated_dir / f"{base_name}_char_{i:03d}.png"
            cv2.imwrite(str(output_path), char_img)

        total_chars += len(chars)

    print(f"\n✅ Isolation completed!")
    print(f"   Total isolated characters: {total_chars}")
    print(f"   Saved in: {isolated_dir}")
    print(f"   Check the folder 'data/isolated/' to verify quality.")

if __name__ == "__main__":
    main()