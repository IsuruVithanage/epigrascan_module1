import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as _ssim
import os
from pathlib import Path



def auto_calibrate(image_path: str) -> dict:
    """
    Automatically measure image characteristics and return optimal parameters.
    Measurements: noise_ratio, contrast, fg_ratio, blob_ratio.
    """
    raw  = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
    H, W = gray.shape[:2]

    lap         = cv2.Laplacian(gray.astype(float), cv2.CV_64F)
    dyn_range   = max(1.0, float(gray.max()) - float(gray.min()))
    noise_ratio = float(np.abs(lap).mean()) / dyn_range
    contrast    = float(np.percentile(gray, 95) - np.percentile(gray, 5))

    _, bw      = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio   = float(np.mean(bw == 255))
    if fg_ratio > 0.55:
        fg_ratio = 1.0 - fg_ratio

    den_quick  = cv2.fastNlMeansDenoising(gray, None, h=15,
                                          templateWindowSize=7, searchWindowSize=21)
    _, bw2     = cv2.threshold(den_quick, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw2 == 255) > 0.55:
        bw2 = cv2.bitwise_not(bw2)
    n2, _, st2, _ = cv2.connectedComponentsWithStats(bw2, connectivity=8)
    areas2     = [int(st2[i, cv2.CC_STAT_AREA]) for i in range(1, n2)]
    blob_ratio = float(max(areas2)) / (H * W) if areas2 else 0.0

    print(f"  Auto-calibrate:  noise_ratio={noise_ratio:.4f}  "
          f"contrast={contrast:.0f}  fg_ratio={fg_ratio:.3f}  "
          f"blob_ratio={blob_ratio:.4f}")

    if noise_ratio < 0.05:    nlm_h = 5
    elif noise_ratio < 0.08:  nlm_h = 10
    elif noise_ratio < 0.12:  nlm_h = 18
    elif noise_ratio < 0.18:  nlm_h = 25
    else:                     nlm_h = 30

    otsu_bias    = 30 if contrast < 150 else 20

    if noise_ratio < 0.06:    noise_thresh = 50
    elif noise_ratio < 0.10:  noise_thresh = 100
    elif noise_ratio < 0.15:  noise_thresh = 150
    else:                     noise_thresh = 200

    if blob_ratio > 0.15:     gap_floor = 0.30
    elif blob_ratio > 0.08:   gap_floor = 0.35
    elif noise_ratio > 0.12:  gap_floor = 0.40
    elif noise_ratio > 0.06:  gap_floor = 0.45
    else:                     gap_floor = 0.55

    params = {
        "nlm_h": nlm_h, "bilateral": True, "otsu_bias": otsu_bias,
        "dilate_kernel": (3, 3), "dilate_iters": 1,
        "white_noise_thresh": noise_thresh, "black_noise_thresh": noise_thresh,
        "gap_floor_ratio": gap_floor, "mzs_threshold": 3.0,
        "_noise_ratio": round(noise_ratio, 4), "_contrast": round(contrast, 1),
        "_fg_ratio": round(fg_ratio, 3), "_blob_ratio": round(blob_ratio, 4),
    }
    print(f"  → nlm_h={nlm_h}  noise_thresh={noise_thresh}  "
          f"gap_floor={gap_floor}  otsu_bias={otsu_bias}")
    return params