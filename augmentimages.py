import os
import cv2
import numpy as np
import random
import albumentations as A
from pathlib import Path

# 1. Configuration
BASE_DIR = '/Users/isuruvithanage/Documents/epigrascan_module1'
input_dir = os.path.join(BASE_DIR, 'data/raw_char')
output_dir = os.path.join(BASE_DIR, 'augmented_char')
num_augmentations = 100

# 2. Augmentation Pipeline (Tuned for Extreme Estampage Degradation)
pipeline = A.Compose([
    A.InvertImg(p=1.0),  # Converts black text to white text

    # Warps straight lines
    A.ElasticTransform(alpha=1.5, sigma=40, p=0.8),

    # Slight rotation and tilt
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-10, 10), shear=(-10, 10), p=0.8),

    # Simulates large missing chunks of stone/ink
    A.CoarseDropout(max_holes=8, max_height=12, max_width=12, min_holes=2, min_height=4, min_width=4, fill_value=0,
                    p=0.7),

    # Extreme noise creates the attached irregular blobs when blurred and thresholded
    A.GaussNoise(var_limit=(300.0, 700.0), p=0.9),

    # Heavier blur to organically merge the noise, dropouts, and main strokes
    A.GaussianBlur(blur_limit=(5, 9), p=0.9)
])

os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_dir):
    print(f"ERROR: Cannot find the input directory at {input_dir}")
else:
    print(f"Scanning directory: {input_dir}")
    folders_processed = 0

    for char_label in os.listdir(input_dir):
        char_path = os.path.join(input_dir, char_label)

        if os.path.isdir(char_path):
            print(f"Processing folder: {char_label}...")
            target_dir = os.path.join(output_dir, char_label)
            os.makedirs(target_dir, exist_ok=True)
            folders_processed += 1

            for img_name in os.listdir(char_path):
                img_path = os.path.join(char_path, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"  - Could not read {img_name}, skipping.")
                    continue

                for i in range(num_augmentations):
                    # Step A: Apply Albumentations (Inversion, Distortion, Dropout, Noise, Blur)
                    augmented = pipeline(image=image)['image']

                    # Step B: Morphological Operations (Using Elliptical Kernels for organic shapes)
                    morph_choice = random.choice(['heavy_blob', 'pitted_erode', 'mixed_degradation', 'none'])

                    if morph_choice == 'heavy_blob':
                        # Creates the very thick, bulbous sections
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        augmented = cv2.dilate(augmented, kernel, iterations=1)

                    elif morph_choice == 'pitted_erode':
                        # Eats away at the edges to simulate worn stone
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        augmented = cv2.erode(augmented, kernel, iterations=1)

                    elif morph_choice == 'mixed_degradation':
                        # Dilate then Erode (Closing) to create highly rounded, blobby terminals
                        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        augmented = cv2.dilate(augmented, kernel_d, iterations=1)
                        augmented = cv2.erode(augmented, kernel_e, iterations=1)

                    # Step C: Final Binarization
                    # This hard-snaps the blurry, noisy image into pure crisp black/white, creating the jagged effect
                    _, binary_img = cv2.threshold(augmented, 127, 255, cv2.THRESH_BINARY)

                    # Save
                    save_path = os.path.join(target_dir, f"{Path(img_name).stem}_aug_{i}.png")
                    cv2.imwrite(save_path, binary_img)

    if folders_processed == 0:
        print("\nWARNING: No subfolders found in 'raw_char'. Images must be inside subfolders.")
    else:
        print("\nAugmentation complete!")