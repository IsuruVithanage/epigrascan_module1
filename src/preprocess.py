import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def morphological_sandwich(image_path: str, config: dict, visualize=False):
    """
    The Morphological Sandwich:
    1. CLAHE (lighting correction)
    2. Adaptive Gaussian Thresholding (binarization)
    3. Dilation (thicken faint strokes)
    4. Median Blur (remove noise)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: CLAHE - Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(
        clipLimit=config['preprocessing']['clahe_clip_limit'],
        tileGridSize=tuple(config['preprocessing']['clahe_grid_size'])
    )
    enhanced = clahe.apply(gray)

    # Step 2: Adaptive Gaussian Thresholding (Inverse because text is usually dark on light background in estampages)
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config['preprocessing']['adaptive_block_size'],
        config['preprocessing']['adaptive_c']
    )

    # Step 3: Dilation - Thickens faint or broken strokes
    kernel = np.ones(tuple(config['preprocessing']['dilation_kernel']), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=config['preprocessing']['dilation_iterations'])

    # Step 4: Median Blur - Removes salt-and-pepper / paper noise while preserving edges
    cleaned = cv2.medianBlur(dilated, config['preprocessing']['median_kernel'])

    if visualize:
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        axs[0].imshow(gray, cmap='gray');
        axs[0].set_title('Original Gray')
        axs[1].imshow(enhanced, cmap='gray');
        axs[1].set_title('CLAHE')
        axs[2].imshow(binary, cmap='gray');
        axs[2].set_title('Binary')
        axs[3].imshow(dilated, cmap='gray');
        axs[3].set_title('Dilated')
        axs[4].imshow(cleaned, cmap='gray');
        axs[4].set_title('Final Cleaned')
        plt.tight_layout()
        plt.show()

    return cleaned


def process_all_images(config):
    input_dir = Path(config['preprocessing']['input_dir'])
    output_dir = Path(config['preprocessing']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.tif"))

    print(f"Found {len(image_paths)} images to preprocess...")

    for img_path in tqdm(image_paths):
        try:
            cleaned = morphological_sandwich(str(img_path), config)
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), cleaned)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")


if __name__ == "__main__":
    config = load_config()
    process_all_images(config)
    print("Preprocessing completed! Check data/preprocessed folder.")