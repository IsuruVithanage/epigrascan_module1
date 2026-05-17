import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def morphological_sandwich(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read {img_path}")

    # Standardize Resolution
    max_dimension = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Convert directly to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # EDIT 1: Decrease Brightness, Decrease Shadows, Increase Black Point
    # Subtracting 50 uniformly darkens the image, crushing the ink dots to black.
    # (Tweak this between 30 and 70 if you need more/less darkness)
    gray = cv2.subtract(gray, 50)

    # EDIT 2: Increase Exposure, Increase Contrast, Increase Highlights
    # Multiplying by 1.6 stretches the surviving text, making it pop bright white.
    # (Tweak this between 1.2 and 2.0 if you need it brighter)
    gray = cv2.convertScaleAbs(gray, alpha=3, beta=10)


    # 1. FORCE THE BACKGROUND TO BLACK (Intensity Cutoff)
    gray[gray < 100] = 0

    # 2. MAKE A BINARY MAP FOR THE COMPUTER
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 3. THE WHITELIST (Only keep the giants)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    for contour in contours:
        area = cv2.contourArea(contour)

        # If the shape is MASSIVE (> 300 pixels), keep it!
        if area > 300:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    # 4. STAMP OUT THE FINAL TEXT
    cleaned_text = cv2.bitwise_and(gray, mask)

    # Convert back to 3-channel BGR for the neural network
    return cv2.cvtColor(cleaned_text, cv2.COLOR_GRAY2BGR)


def main():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_estampages"])
    proc_dir = Path(config["paths"]["processed"])

    proc_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(raw_dir.glob("*.[jp][pn]*"))
    print(f"Found {len(image_files)} images to process...\n")

    for img_path in tqdm(image_files, desc="Preprocessing images"):
        try:
            processed_img = morphological_sandwich(img_path)
            out_path = proc_dir / f"{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), processed_img)
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")

    print(f"\n✅ Preprocessing completed! Cleaned images saved in: {proc_dir}")


if __name__ == "__main__":
    main()