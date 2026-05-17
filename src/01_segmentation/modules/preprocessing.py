import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as _ssim
import os
from pathlib import Path

from .baseline import detect_baseline



def preprocess(raw_bgr,
               nlm_h: int         = 10,
               bilateral: bool    = True,
               otsu_bias: int     = 20,
               dilate_kernel: tuple = (3, 3),
               dilate_iters: int  = 1):
    """
    Preprocessing pipeline.

    Parameters (all tunable per image)
    ------------------------------------
    nlm_h          : Non-Local Means filter strength. Higher = more smoothing
                     but risks erasing thin strokes. Default 10 (gentle).
                     Reference paper uses h=30 — heavier, but they also add
                     bilateral filtering afterwards to recover edges.
    bilateral      : If True, apply Bilateral Filter after NLMeans (as the
                     reference paper does). Bilateral preserves character edges
                     while smoothing flat regions. d=9, sigmaColor=sigmaSpace=75.
    otsu_bias      : Subtracted from Otsu's global threshold before binarising.
                     Higher bias → more aggressive (dark ink) capture.
                     Default 20 (from paper). Lower for faded inscriptions.
    dilate_kernel  : (w, h) of structuring element for the stroke-gap-closing
                     morphological closing step (reference paper: configurable).
                     Default (3,3) ellipse. Increase for heavily fragmented strokes.
    dilate_iters   : Iterations for the morphological closing. Default 1.

    Returns
    -------
    gray, binary_clean, dilated_crop, metrics

    metrics dict contains PSNR, SSIM, Laplacian variance, and edge retention
    ratio — the four quality indicators reported by the reference paper.
    """
    gray = (cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
            if raw_bgr.ndim == 3 else raw_bgr.copy())

    # ── Step 1: Non-Local Means denoising ────────────────────────────────
    den = cv2.fastNlMeansDenoising(gray, None, h=nlm_h,
                                   templateWindowSize=7, searchWindowSize=21)

    # ── Step 2 (NEW): Bilateral filter — edge-preserving second pass ─────
    # The reference paper applies this after NLMeans to further smooth noise
    # in flat background regions without blurring character stroke boundaries.
    # Key advantage for Brahmi: the bilateral kernel's range-σ (75) keeps sharp
    # ink edges intact while the spatial-σ (75) smooths surface texture noise.
    if bilateral:
        den = cv2.bilateralFilter(den, d=9, sigmaColor=75, sigmaSpace=75)

    # ── Step 3: Adjusted Otsu binarisation ───────────────────────────────
    t_glob, _ = cv2.threshold(den, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_adj = max(0, t_glob - otsu_bias)
    _, binary = cv2.threshold(den, t_adj, 255, cv2.THRESH_BINARY)
    print(f"  Otsu T_global={t_glob:.0f}  T_adjusted={t_adj:.0f}  "
          f"(bilateral={'on' if bilateral else 'off'}  nlm_h={nlm_h})")

    # Auto-polarity: ensure WHITE = foreground text
    if np.mean(binary == 255) > 0.55:
        binary = cv2.bitwise_not(binary)
        print("  Polarity: inverted (background was white)")
    else:
        print("  Polarity: kept (background already black)")

    # ── Step 4: Morphological closing — configurable kernel/iterations ───
    # Reference paper: user-tunable kernel size and iteration count so
    # researchers can adapt to varying inscription quality / DPI.
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_kernel)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se,
                                    iterations=dilate_iters)

    # Large dilation for crop-region detection ONLY (unchanged)
    se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    dilated_crop = cv2.dilate(binary_clean, se_h, iterations=1)
    dilated_crop = cv2.dilate(dilated_crop, se_v, iterations=1)

    # ── Quality metrics (reference paper Section 7.1.1) ──────────────────
    # Computed once here so run_module1 can log and save them.
    mse = np.mean((gray.astype(float) - den.astype(float)) ** 2)
    psnr = float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 99.0

    ssim_val, _ = _ssim(gray, den, full=True)

    lap_var = float(cv2.Laplacian(binary_clean, cv2.CV_64F).var())

    edges_orig = cv2.Canny(gray, 50, 150)
    edges_proc = cv2.Canny(den,  50, 150)
    n_orig = int(np.sum(edges_orig > 0))
    edge_retention = (float(np.sum(edges_proc > 0)) / n_orig
                      if n_orig > 0 else 1.0)

    metrics = {
        "psnr":            round(psnr,  2),
        "ssim":            round(float(ssim_val), 4),
        "laplacian_var":   round(lap_var, 1),
        "edge_retention":  round(edge_retention, 3),
    }
    print(f"  Quality metrics: PSNR={psnr:.1f}dB  SSIM={ssim_val:.3f}  "
          f"Laplacian={lap_var:.0f}  EdgeRetention={edge_retention*100:.1f}%")

    return gray, binary_clean, dilated_crop, metrics

def crop_to_inscription(dilated_crop, binary_clean, pad=8):
    """
    Find the inscription bounding box from the dilated binary and crop
    the clean binary to that region.

    ROOT CAUSE FIX (union-bbox):
    ─────────────────────────────
    The old code picked only the SINGLE LARGEST connected component.
    This fails when the inscription contains multiple character groups
    with dark stone texture between them — the right group may be larger
    than the left group (more pixels due to noise accumulation), so the
    left characters are silently dropped.

    Fix: use the UNION bounding box of ALL significant blobs.
    A blob is 'significant' if its area ≥ min_area_ratio of the image area.
    This guarantees every character cluster is included regardless of which
    individual blob happens to be the largest.

    min_area_ratio is set to 0.5% of image area — large enough to ignore
    isolated dust specks but small enough to catch small character groups.
    """
    H, W = binary_clean.shape[:2]
    nl, _, st, _ = cv2.connectedComponentsWithStats(dilated_crop, connectivity=8)

    if nl < 2:
        return binary_clean.copy(), (0, 0, W, H)

    img_area     = H * W
    min_blob_area = max(50, img_area * 0.005)   # 0.5% of image area

    # Collect all significant blobs
    sig_blobs = [
        (int(st[i, cv2.CC_STAT_LEFT]),
         int(st[i, cv2.CC_STAT_TOP]),
         int(st[i, cv2.CC_STAT_WIDTH]),
         int(st[i, cv2.CC_STAT_HEIGHT]))
        for i in range(1, nl)
        if st[i, cv2.CC_STAT_AREA] >= min_blob_area
    ]

    if not sig_blobs:
        # Fallback: use entire image
        return binary_clean.copy(), (0, 0, W, H)

    # Union bounding box
    x_min = min(b[0] for b in sig_blobs)
    y_min = min(b[1] for b in sig_blobs)
    x_max = max(b[0] + b[2] for b in sig_blobs)
    y_max = max(b[1] + b[3] for b in sig_blobs)

    x0 = max(0, x_min - pad)
    y0 = max(0, y_min - pad)
    x1 = min(W, x_max + pad)
    y1 = min(H, y_max + pad)

    n_blobs = len(sig_blobs)
    print(f"  Crop: {n_blobs} significant blob(s) → "
          f"union bbox x={x0}-{x1}  y={y0}-{y1}")

    return binary_clean[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

def extract_character_band(binary_img, padding: int = 12,
                           band_height_factor: float = 1.3,
                           save_vis_path: str = None) -> tuple:
    """
    Permanently remove noise outside the character text band by masking
    pixels that lie above/below the fitted inscription baseline.

    Root cause of the old failure
    ──────────────────────────────
    The previous sliding-window "peak-row" approach chose the row with the
    MOST white pixels per column strip. For inscriptions with a solid stone
    texture band at the bottom, that band always wins — it has more pixels
    per row than the sparse character strokes above it.

    Correct approach
    ─────────────────
    Use detect_baseline() which fits a polynomial to CCA component CENTROIDS
    filtered by area. This correctly identifies the character region because:
      • Character components have intermediate areas (not tiny specks,
        not the massive connected stone blob)
      • The polynomial fit through their centroids follows the text flow
      • It is immune to the solid noise band which appears as one giant
        component (area >> threshold) and gets excluded

    Algorithm
    ─────────
    1. Run detect_baseline on the noisy binary → get baseline polynomial.
    2. Estimate character half-height from CCA components whose centroids
       are NEAR the baseline (within ±30px), excluding outliers.
    3. For every column x:
         allowed_y = [baseline(x) − half_h − padding,
                      baseline(x) + half_h + padding]
       Zero all pixels outside this range.
    4. Optionally save a debug visualisation.

    Parameters
    ──────────
    binary_img          : WHITE text on BLACK background
    padding             : extra margin above/below band in pixels (default 12)
    band_height_factor  : band_half = median_char_height × this factor (1.3)
    save_vis_path       : path to save green-spline / red-boundary debug image

    Returns
    ───────
    (masked_binary, baseline_info, band_half)
    """
    H, W = binary_img.shape[:2]

    # ── Step 1: detect_baseline gives us the correct band centre ─────────
    baseline = detect_baseline(binary_img)
    poly     = baseline["poly"]

    # ── Step 2: estimate character height from NEAR-BASELINE components ──
    n, lbl, st, cent = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8)

    img_area   = H * W
    near_heights = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        # Skip too-small (noise specks) and too-large (noise blob)
        if area < 30 or area > img_area * 0.03:
            continue
        cy    = float(cent[i][1])
        cx    = float(cent[i][0])
        bl_y  = float(poly(cx))
        dist  = abs(cy - bl_y)
        if dist < 40:                           # within 40px of baseline
            near_heights.append(int(st[i, cv2.CC_STAT_HEIGHT]))

    if near_heights:
        med_h = float(np.median(near_heights))
    else:
        # Fallback: use all character-scale component heights
        all_h = [int(st[i, cv2.CC_STAT_HEIGHT])
                 for i in range(1, n)
                 if 30 <= int(st[i, cv2.CC_STAT_AREA]) <= img_area * 0.03]
        med_h = float(np.median(all_h)) if all_h else H * 0.25

    band_half = max(int(med_h * band_height_factor),
                    int(H * 0.18))   # at least 18% of image height

    print(f"  Band: baseline_flow={baseline['flow_type']}  "
          f"median_char_h={med_h:.0f}px  "
          f"band_half={band_half}px  total={2*(band_half+padding)}px  "
          f"image_H={H}px")

    # ── Step 3: apply band mask ───────────────────────────────────────────
    mask = np.zeros((H, W), dtype=np.uint8)
    for col in range(W):
        cy = int(np.clip(float(poly(col)), 0, H - 1))
        y0 = max(0, cy - band_half - padding)
        y1 = min(H, cy + band_half + padding)
        mask[y0:y1, col] = 255

    masked = cv2.bitwise_and(binary_img, mask)

    n_before = cv2.connectedComponentsWithStats(binary_img, connectivity=8)[0] - 1
    n_after  = cv2.connectedComponentsWithStats(masked,     connectivity=8)[0] - 1
    print(f"  Noise components removed: {n_before - n_after}  "
          f"({n_before} → {n_after})")

    # ── Step 4: optional debug visualisation ─────────────────────────────
    if save_vis_path:
        vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for col in range(W - 1):
            cy  = int(np.clip(float(poly(col)),     0, H - 1))
            cy2 = int(np.clip(float(poly(col + 1)), 0, H - 1))
            # Spline baseline (green)
            cv2.line(vis, (col, cy), (col + 1, cy2), (0, 220, 0), 2)
            # Band boundaries (red)
            y0 = max(0, cy - band_half - padding)
            y1 = min(H - 1, cy + band_half + padding)
            if 0 <= y0 < H: vis[y0, col] = [0, 0, 255]
            if 0 <= y1 < H: vis[y1, col] = [0, 0, 255]
        cv2.imwrite(save_vis_path, vis)
        print(f"  Band vis → {save_vis_path}")

    return masked, baseline, band_half

def _cca_filter(img, threshold, kill_color, replace_color):
    mask = img if kill_color == 255 else cv2.bitwise_not(img)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = img.copy()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < threshold:
            out[lbl == i] = replace_color
    return out

def noise_removal(cropped, white_thresh=200, black_thresh=200):
    """
    Two-pass CCA noise removal.  Input/output: WHITE text on BLACK background.

    Pass 1 (on white-on-black directly):
        Find small WHITE clusters (< white_thresh px²).
        These are small foreground noise specks and micro char fragments.
        Convert them to BLACK → closes minor gaps, cleans speckle noise.

    Invert → now BLACK chars on WHITE background.

    Pass 2 (on the inverted black-on-white image):
        Find small BLACK clusters (< black_thresh px²).
        In the inverted image, BLACK = characters.
        Small black clusters = isolated char fragment noise.
        Convert them to WHITE → erases tiny isolated char noise specks.

        BUG NOTE fixed here: Pass 2 must use kill_color=0 (kill BLACK),
        NOT kill_color=255 (which would kill white background fragments —
        the opposite of what is needed).  The reference paper Ch.5 states:
        "black pixel clusters now representing residual noise or artifacts
        after inversion. Small black clusters are converted to white."

    Re-invert → WHITE chars on BLACK, clean.
    """
    # Pass 1: remove small WHITE (foreground noise) from white-on-black
    p1  = _cca_filter(cropped, white_thresh, kill_color=255, replace_color=0)

    # Invert to black-on-white for pass 2
    inv = cv2.bitwise_not(p1)

    # Pass 2: remove small BLACK (char fragment noise) from black-on-white
    #         kill_color=0  → mask = bitwise_not(inv) = highlights black regions
    #         replace_color=255 → erase found black clusters to white
    p2  = _cca_filter(inv, black_thresh, kill_color=0, replace_color=255)

    # Re-invert to restore white-on-black
    return p1, inv, cv2.bitwise_not(p2)

def remove_border_blobs(binary_img):
    """
    Slices off huge border noise while perfectly preserving letters.
    Uses a morphological 'opening' trick to sever thin bridges between
    characters and the border noise, then deletes the isolated border chunks.
    """
    H, W = binary_img.shape[:2]

    # Step 1: Break the thin bridges connecting text to the borders
    # A 5x5 kernel is usually perfect for snapping these false connections.
    sever_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, sever_kernel, iterations=1)

    # Step 2: Run connected components on the OPENED (disconnected) image
    n, lbl, st, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    # Step 3: Build a mask of ONLY the massive border noise
    noise_mask = np.zeros((H, W), dtype=np.uint8)
    removed_count = 0

    for i in range(1, n):
        x = int(st[i, cv2.CC_STAT_LEFT])
        y = int(st[i, cv2.CC_STAT_TOP])
        w = int(st[i, cv2.CC_STAT_WIDTH])
        h = int(st[i, cv2.CC_STAT_HEIGHT])
        area = st[i, cv2.CC_STAT_AREA]

        touches_border = (x == 0 or y == 0 or x + w >= W or y + h >= H)

        # Identify as noise if it touches the edge AND:
        # A) It is confined to the top/bottom margins, OR
        # B) It is a massive heavy block (e.g., > 4% of total image area)
        is_marginal = (y + h < H * 0.35) or (y > H * 0.65)
        is_huge_block = area > (H * W * 0.04)

        if touches_border and (is_marginal or is_huge_block):
            noise_mask[lbl == i] = 255
            removed_count += 1

    # Step 4: Re-expand the noise mask slightly.
    # Because 'opening' shrank the noise, we dilate the mask so it completely
    # covers the noise footprint in the original image.
    noise_mask = cv2.dilate(noise_mask, sever_kernel, iterations=2)

    # Step 5: Subtract the noise mask from the ORIGINAL, un-shrunk binary image.
    # This leaves your letters pristine and un-degraded!
    out = cv2.bitwise_and(binary_img, cv2.bitwise_not(noise_mask))

    print(f"  Border blob removal: Severed and removed {removed_count} massive border chunks")
    return out