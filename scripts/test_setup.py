import cv2
import numpy as np
from pathlib import Path

print("✅ Project initialized successfully!")

# Test 1: Check folders
print("Raw estampages found:", len(list(Path("data/raw_estampages").glob("*.jpg"))) + len(list(Path("data/raw_estampages").glob("*.png"))))
print("Kaggle classes found:", len(list(Path("data/kaggle_brahmi").glob("*"))))

# Test 2: Load one image
if list(Path("data/raw_estampages").glob("*.jpg")):
    test_img = cv2.imread(str(next(Path("data/raw_estampages").glob("*.jpg"))))
    print("✅ Can read estampage image. Shape:", test_img.shape)
else:
    print("No estampage yet - add one to test.")

print("\n🎉 Ready for Step 2 (Morphological Sandwich)!")