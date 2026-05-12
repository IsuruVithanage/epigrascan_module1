"""
Module 1: Image Preprocessing and Character Segmentation
for Ancient Brahmi Inscription Estampages
=========================================================

Design goal
-----------
Accurate CHARACTER COUNT is the primary objective.  The language model in
Module 3 needs an exact count; a single missed or phantom character breaks
the restoration chain.

New architecture
----------------
Step 1  Preprocessing      grayscale → NLMeans → Otsu(T-20) → invert → polish
Step 2  Crop               large-kernel dilation used only for bounding-box
                           detection; character work always uses clean binary
Step 3  Noise removal      two-pass CCA size filtering
Step 4  Baseline analysis  fit polynomial to character centroids; classify
                           flow as STRAIGHT or CURVED; measure curvature
Step 5  Rectification      straighten curved text onto a horizontal band
                           (identity transform for straight text)
Step 6  Multi-signal count three independent estimators vote on character count
                             A) vertical projection profile valleys
                             B) CCA component widths vs expected width
                             C) inter-gap histogram (gap vs stroke peaks)
                           Weighted majority -> final count N
Step 7  Boundary placement place N-1 cut lines at lowest projection valleys
Step 8  Width validation   Modified Z-Score flags anomalous segments; each
                           flagged segment is re-split once
Step 9  Crop & export      crop each character from the clean binary image
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as _ssim
import os
from pathlib import Path


# ======================================================================
# 1.  PREPROCESSING
# ======================================================================

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


# ======================================================================
# 2.  CROP TO INSCRIPTION REGION
# ======================================================================

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


# ======================================================================
# 3b.  CHARACTER BAND EXTRACTION  (spline-guided noise removal)
# ======================================================================

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


# ======================================================================
# 4.  BASELINE DETECTION AND FLOW CLASSIFICATION
# ======================================================================

def _get_char_centroids(binary):
    """CCA -> list of component dicts sorted by x-centroid."""
    n, lbl, st, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        comps.append({
            "cx": float(cent[i][0]), "cy": float(cent[i][1]),
            "x":  int(st[i, cv2.CC_STAT_LEFT]),
            "y":  int(st[i, cv2.CC_STAT_TOP]),
            "w":  int(st[i, cv2.CC_STAT_WIDTH]),
            "h":  int(st[i, cv2.CC_STAT_HEIGHT]),
            "area": int(area),
        })
    comps.sort(key=lambda c: c["cx"])
    return comps


def _cluster_centroids_by_row(chars, H):
    """
    Split centroids into row-groups using KDE valley detection.

    Why KDE instead of sorted-gap detection
    ─────────────────────────────────────────
    Simple gap detection (find large gaps in sorted y values) fails when
    two text rows are CLOSE together — the gap between rows may be smaller
    than gaps within one wavy row.

    KDE approach:
    1. Build a histogram of centroid y-values and smooth with Gaussian σ=6px.
    2. Find VALLEYS in the KDE — valleys in the y-density are row separators.
    3. Split centroids at each valley.
    4. Sort groups by total area (dominant / most-character row first).

    If no valley is found the single group is returned unchanged.
    """
    if not chars:
        return [chars]

    # Build and smooth y-histogram
    y_hist = np.zeros(H)
    for c in chars:
        idx = int(np.clip(c["cy"], 0, H - 1))
        y_hist[idx] += 1
    kde = gaussian_filter1d(y_hist, sigma=6)

    if kde.max() == 0:
        return [chars]

    # Find valleys in KDE = row separator positions
    inv_kde     = kde.max() - kde
    valleys, _  = find_peaks(
        inv_kde,
        height=kde.max() * 0.25,       # valley must dip to < 75% of peak
        distance=max(10, int(H * 0.10)),
        prominence=kde.max() * 0.15,   # valley must be prominent
    )

    if len(valleys) == 0:
        return [chars]   # single row — no separator found

    # Split centroids at each valley
    separators = sorted(int(v) for v in valleys)
    groups     = []
    prev_sep   = -1
    for sep in separators + [H + 1]:
        group = [c for c in chars if prev_sep < c["cy"] <= sep]
        if group:
            groups.append(group)
        prev_sep = sep

    if not groups:
        return [chars]

    # Sort by total area descending (dominant row first)
    groups.sort(key=lambda g: sum(c["area"] for c in g), reverse=True)

    print(f"  Baseline row-split: KDE found {len(valleys)} valley(s) at "
          f"y={separators} → {len(groups)} row group(s)  "
          f"sizes={[len(g) for g in groups]}")
    return groups


def detect_baseline(binary_img):
    """
    Fit a polynomial baseline through ALL character centroids.

    Key fix (matching Phoenix team approach): fit ALL centroids first.
    For a curved inscription, centroids span the full y-range of the wave.
    Pre-splitting by row discards the dip/rise centroids that DEFINE the
    curve, leaving only flat middle points → wrong straight line.
    Row-splitting is only used as fallback when residual > 22% image height.
    """
    H, W  = binary_img.shape[:2]
    comps = _get_char_centroids(binary_img)
    areas = [c["area"] for c in comps]
    if not areas:
        return _fallback_baseline(W, H)
    thresh = max(50, float(np.percentile(areas, 30)))
    chars  = [c for c in comps if c["area"] >= thresh]
    if len(chars) < 3:
        return _fallback_baseline(W, H)

    xs = np.array([c["cx"] for c in chars])
    ys = np.array([c["cy"] for c in chars])

    def _fit_and_select(xs_, ys_, label=""):
        fits_ = {}
        for deg in range(1, min(4, len(xs_))):
            cf   = np.polyfit(xs_, ys_, deg)
            poly = np.poly1d(cf)
            xs_f = np.linspace(0, W, 300)
            ys_f = poly(xs_f)
            fits_[deg] = {"cf": cf, "poly": poly,
                          "std":  float(np.std(ys_ - poly(xs_))),
                          "curv": float(ys_f.max() - ys_f.min())}
        for d in [1, 2, 3]:
            if d not in fits_:
                fits_[d] = fits_[max(fits_.keys())]
        def _up(lo, hi):
            sl,sh = fits_[lo]["std"], fits_[hi]["std"]
            cl,ch = fits_[lo]["curv"],fits_[hi]["curv"]
            return ((sl-sh)/max(sl,1e-6)*100 >= 10.0 or
                    (ch-cl >= 12.0 and sh <= sl*1.05))
        ch = 1
        if _up(1,2): ch = 2
        if ch==2 and _up(2,3): ch = 3
        if label:
            print(f"  Baseline{label}: deg{ch}  "
                  f"std={fits_[ch]['std']:.1f}px  curv={fits_[ch]['curv']:.1f}px")
        return fits_, ch

    # Step 1: fit ALL centroids
    fits, chosen = _fit_and_select(xs, ys, " (all pts)")

    # Step 2: RANSAC using chosen polynomial (not linear!)
    best_poly = fits[chosen]["poly"]
    inlier    = np.abs(ys - best_poly(xs)) <= max(20.0, fits[chosen]["std"]*2.0)
    if inlier.sum() >= 3 and inlier.sum() < len(xs):
        print(f"  Baseline RANSAC: removed {(~inlier).sum()} outlier(s), "
              f"kept {inlier.sum()}/{len(xs)}")
        xs, ys = xs[inlier], ys[inlier]
        fits, chosen = _fit_and_select(xs, ys, " (after RANSAC)")

    # Step 3: row-split fallback (only if residual still very large)
    if fits[chosen]["std"] > H * 0.22 and len(chars) >= 6:
        print(f"  Baseline: std too large -> trying row-split fallback")
        row_groups = _cluster_centroids_by_row(chars, H)
        if len(row_groups) > 1 and len(row_groups[0]) >= 3:
            xs_d = np.array([c["cx"] for c in row_groups[0]])
            ys_d = np.array([c["cy"] for c in row_groups[0]])
            fits_d, ch_d = _fit_and_select(xs_d, ys_d, " (dominant row)")
            if fits_d[ch_d]["std"] < fits[chosen]["std"] * 0.85:
                fits, chosen, xs, ys = fits_d, ch_d, xs_d, ys_d
                print("  Baseline: row-split improved fit -> using dominant row")

    poly = fits[chosen]["poly"]
    std  = fits[chosen]["std"]
    curv = fits[chosen]["curv"]
    xs_f = np.linspace(0, W, 300)
    ys_f = poly(xs_f)

    if chosen == 1 and std < 0.05*H: flow = "straight"
    elif curv < 0.08*H:              flow = "straight"
    elif chosen == 2:                 flow = "curved"
    else:                             flow = "wavy"

    angle = (float(np.degrees(np.arctan2(
                float(ys[-1])-float(ys[0]),
                float(xs[-1])-float(xs[0]))))
             if len(xs) >= 2 else 0.0)

    print(f"  Baseline: flow={flow}  degree={chosen}  "
          f"curvature={curv:.1f}px  residual_std={std:.1f}px  "
          f"tilt={angle:.1f}deg  pts_used={len(xs)}")

    return {"flow_type": flow, "poly": poly,
            "poly_coeffs": fits[chosen]["cf"],
            "degree": chosen, "residual_std": std, "curvature": curv,
            "angle_deg": angle,
            "centroids": list(zip(xs.tolist(), ys.tolist())),
            "image_W": W, "image_H": H}



def _fallback_baseline(W, H):
    poly = np.poly1d([H / 2])
    return {"flow_type": "straight", "poly": poly, "poly_coeffs": [H/2],
            "degree": 0, "residual_std": 0.0, "curvature": 0.0,
            "angle_deg": 0.0, "centroids": [], "image_W": W, "image_H": H}


# ======================================================================
# 5.  RECTIFICATION  (straighten curved text)
# ======================================================================

def rectify(binary_img, baseline_info):
    """
    For CURVED/WAVY: shift each column vertically so baseline sits at
    a constant y.  For STRAIGHT: return image as-is.

    Returns (rect_img, col_offsets)
    """
    H, W  = binary_img.shape[:2]
    flow  = baseline_info["flow_type"]
    poly  = baseline_info["poly"]

    if flow == "straight":
        return binary_img.copy(), np.zeros(W, dtype=int)

    target_y    = int(poly(W / 2))
    col_offsets = np.zeros(W, dtype=int)
    rect_img    = np.zeros_like(binary_img)

    for col in range(W):
        shift = int(poly(col)) - target_y
        col_offsets[col] = shift
        src = binary_img[:, col]
        if shift > 0:
            rect_img[:H-shift, col] = src[shift:]
        elif shift < 0:
            rect_img[-shift:, col]  = src[:shift]
        else:
            rect_img[:, col] = src

    return rect_img, col_offsets


# How low the projection must dip for a valley to be a real inter-character gap.
# 0.45 means the projection must fall to 45% or less of the max at that column.
# RAISE this (e.g. 0.60) for cleaner images with well-separated characters.
# LOWER this (e.g. 0.30) for dense/damaged inscriptions with shallow gaps.
GAP_FLOOR_RATIO = 0.45

def _smooth_proj(binary_img, sigma=3):
    proj = np.sum(binary_img == 255, axis=0).astype(float)
    return gaussian_filter1d(proj, sigma=sigma)


def _estimate_min_char_width(binary_img, W):
    """
    Estimate the minimum expected character width from CCA.
    Uses the 20th percentile of character-scale component widths.
    This is the crucial reference for rejecting intra-character valleys.
    """
    comps = _get_char_centroids(binary_img)
    if not comps:
        return max(10, int(W * 0.04))

    areas = [c["area"] for c in comps]
    thresh = float(np.percentile(areas, 35))
    chars  = [c for c in comps if c["area"] >= thresh]
    if not chars:
        return max(10, int(W * 0.04))

    widths   = sorted(c["w"] for c in chars)
    # Use 20th percentile * 0.55 as the absolute floor.
    # A valid character segment must be at least this wide.
    min_w = max(8, int(np.percentile(widths, 20) * 0.55))
    return min_w


def _estimator_projection(proj_s, W, binary_img, min_w_frac=0.035):
    """
    Estimator A: count INTER-CHARACTER valleys.

    TWO-LAYER filter to reject intra-character dips:

    Layer 1 – Absolute floor:
      A real inter-character gap drops the projection to near zero.
      An intra-character hollow (e.g. inside a square character) or a
      bridging stroke NEVER reaches zero — the valley stays above the
      noise floor.  Cut threshold: valley must be below
      GAP_FLOOR_RATIO * max_projection.  Any deeper valleys ARE real gaps.
      Any shallower valleys are internal structure, not boundaries.

    Layer 2 – Minimum segment width:
      Even if a valley passes the floor check, it is rejected if it would
      create a segment narrower than one estimated character width.

    Returns (count, valley_cols).
    """
    GAP_FLOOR_RATIO = globals().get("GAP_FLOOR_RATIO", 0.45)

    min_dist   = max(5, int(W * min_w_frac))
    min_char_w = _estimate_min_char_width(binary_img, W)
    print(f"    min_char_width estimate = {min_char_w}px  "
          f"gap_floor = {GAP_FLOOR_RATIO*100:.0f}% of max")

    nonzero = np.where(proj_s > proj_s.max() * 0.04)[0]
    if len(nonzero) < 4:
        return 1, []
    t0, t1 = int(nonzero[0]), int(nonzero[-1])
    span    = proj_s[t0:t1]
    if span.max() == 0:
        return 1, []

    # Absolute gap floor: anything above this is intra-character structure
    gap_floor = span.max() * GAP_FLOOR_RATIO

    inv = span.max() - span
    all_peaks, _ = find_peaks(inv,
                              prominence=span.max() * 0.20,
                              distance=min_dist)

    if len(all_peaks) == 0:
        return 1, []

    # Layer 1: keep only valleys where proj_s actually dips below gap_floor
    floor_passed = [pk for pk in all_peaks
                    if proj_s[pk + t0] <= gap_floor]

    raw_valleys = sorted(int(pk) + t0 for pk in floor_passed)

    if not raw_valleys:
        # No valley dips low enough → all characters are joined; return 1
        # (caller will handle via equal-spacing fallback)
        print(f"    No valleys pass floor check — inscription band fully joined")
        return 1, []

    # Layer 2: minimum segment width constraint
    valid_valleys = _filter_valleys_by_min_width(
        raw_valleys, t0, t1, min_char_w, proj_s
    )

    return len(valid_valleys) + 1, valid_valleys


def _filter_valleys_by_min_width(valleys, t0, t1, min_char_w, proj_s):
    """
    Iteratively remove the shallowest valley that produces a too-narrow
    segment, until all resulting segments meet the minimum width.

    Uses a greedy approach: at each iteration, if any segment is too narrow,
    remove the valley that created it (the one with the highest projection
    value, i.e. the weakest cut).
    """
    valleys = sorted(valleys)
    changed = True
    while changed:
        changed = False
        boundaries = [t0] + valleys + [t1]
        seg_widths = [boundaries[i+1] - boundaries[i]
                      for i in range(len(boundaries)-1)]

        # Find segments that are too narrow
        too_narrow = [i for i, sw in enumerate(seg_widths) if sw < min_char_w]
        if not too_narrow:
            break

        # For each too-narrow segment, find the valley responsible
        # (the boundary on its left or right that is a valley, not t0/t1)
        # Remove the weakest valley adjacent to a too-narrow segment
        candidate_removals = set()
        for seg_idx in too_narrow:
            # Left boundary of this segment
            left_b  = boundaries[seg_idx]
            # Right boundary of this segment
            right_b = boundaries[seg_idx + 1]
            if left_b in valleys:
                candidate_removals.add(left_b)
            if right_b in valleys:
                candidate_removals.add(right_b)

        if not candidate_removals:
            break

        # Remove the candidate with the highest projection value (weakest gap)
        weakest = max(candidate_removals, key=lambda v: proj_s[v])
        valleys.remove(weakest)
        changed = True

    return valleys


def _estimator_cca(binary_img):
    """
    Estimator B: count character-scale CCA components.

    Robust approach:
    - Filter out noise (area < 5th percentile) and over-merged blobs
    - Estimate expected char width using the MODE of the width histogram
      (mode is immune to the long tail of merged blobs)
    - Divide each blob width by expected_char_width and round to get
      how many chars it contains
    """
    comps = _get_char_centroids(binary_img)
    if not comps:
        return 1

    areas  = np.array([c["area"] for c in comps])
    a_lo   = float(np.percentile(areas, 10))
    a_hi   = float(np.percentile(areas, 90))
    # keep only components in the 10th–90th percentile area range
    chars  = [c for c in comps if a_lo <= c["area"] <= a_hi * 3]
    if not chars:
        chars = comps  # fallback to all

    widths = np.array([c["w"] for c in chars])
    if len(widths) == 0:
        return 1

    # Mode via histogram (10-bin) — robust against outliers
    hist, edges = np.histogram(widths, bins=min(10, len(widths)))
    modal_bin   = np.argmax(hist)
    modal_w     = float((edges[modal_bin] + edges[modal_bin+1]) / 2)
    if modal_w < 3:
        modal_w = float(np.median(widths))

    total = 0
    for c in chars:
        n = max(1, round(c["w"] / modal_w))
        total += n
    return max(1, int(total))


def _estimator_gaps(proj_s):
    """
    Estimator C: count runs of near-zero projection columns (gap segments).
    """
    thresh = proj_s.max() * 0.08
    is_gap = (proj_s < thresh).astype(int)
    in_gap, gap_count, run = False, 0, 0
    for v in is_gap:
        if v:
            run += 1; in_gap = True
        else:
            if in_gap and run >= 2:
                gap_count += 1
            in_gap = False; run = 0
    if in_gap and run >= 2:
        gap_count += 1
    return max(1, gap_count + 1)


def count_characters(binary_img, baseline_info):
    """
    Run three estimators and combine by tolerance-based voting.

    Outlier rejection: before voting, discard any estimator whose value
    differs from the median of the three by more than 50%.  This prevents
    a wildly wrong estimator (e.g. CCA returning 81 due to fragmentation)
    from distorting the final count.

    Tolerance: two counts that differ by <= 1 are considered agreeing.

    Confidence:
      high   - all three agree
      medium - two agree
      low    - none agree (use projection, most robust single estimator)

    Returns (count, confidence, detail_dict)
    """
    W      = binary_img.shape[1]
    proj_s = _smooth_proj(binary_img)

    cnt_a, valleys = _estimator_projection(proj_s, W, binary_img)
    cnt_b          = _estimator_cca(binary_img)
    cnt_c          = _estimator_gaps(proj_s)

    print(f"  Estimator A (projection): {cnt_a}")
    print(f"  Estimator B (CCA):        {cnt_b}")
    print(f"  Estimator C (gap-runs):   {cnt_c}")

    # Outlier rejection — discard any estimator > 50% from the median
    raw = np.array([cnt_a, cnt_b, cnt_c], dtype=float)
    med = np.median(raw)
    valid = {name: v for name, v in zip(["A","B","C"], [cnt_a, cnt_b, cnt_c])
             if med == 0 or abs(v - med) / med <= 0.50}

    if len(valid) < 2:
        # All diverge wildly — trust projection only
        final, conf = cnt_a, "low"
        print(f"  Outlier rejection: all estimators diverge, using projection")
    else:
        vals = list(valid.values())
        names = list(valid.keys())
        print(f"  Valid estimators after outlier rejection: "
              f"{dict(zip(names, vals))}")

        def near(a, b): return abs(a - b) <= 1

        if len(vals) == 3:
            a, b, c = vals
            ab, ac, bc = near(a,b), near(a,c), near(b,c)
            if ab and ac:
                final, conf = a, "high"
            elif ab:
                final, conf = round((a + b) / 2), "medium"
            elif ac:
                final, conf = round((a + c) / 2), "medium"
            elif bc:
                final, conf = round((b + c) / 2), "medium"
            else:
                final, conf = cnt_a, "low"
        else:
            a, b = vals[0], vals[1]
            if near(a, b):
                final, conf = round((a + b) / 2), "medium"
            else:
                # Use the projection estimator if available, else lower of two
                final = cnt_a if "A" in valid else min(a, b)
                conf  = "low"

    print(f"  => Final count = {final}  confidence = {conf}")
    return final, conf, {"proj": cnt_a, "cca": cnt_b, "gaps": cnt_c,
                         "valleys": valleys, "proj_s": proj_s}


# ======================================================================
# 7.  BOUNDARY PLACEMENT
# ======================================================================

def place_boundaries(proj_s, n_chars, image_W, binary_img):
    """
    Select exactly n_chars-1 cut positions.

    Priority order for cut selection:
    1. Valleys that pass the gap-floor check (proj < 40% of max) — sorted by depth
    2. If fewer floor-passing valleys than needed: supplement with
       the lowest projection points in each inter-character zone
    3. Fallback: equal-spacing biased toward local minima

    This ensures cuts land in genuine gaps, not inside characters.
    """
    W      = image_W
    n_cuts = n_chars - 1

    nonzero = np.where(proj_s > proj_s.max() * 0.04)[0]
    t0 = int(nonzero[0])  if len(nonzero) else 0
    t1 = int(nonzero[-1]) if len(nonzero) else W

    if n_cuts <= 0:
        return [t0, t1]

    gap_floor  = proj_s[t0:t1].max() * globals().get("GAP_FLOOR_RATIO", 0.45)
    min_char_w = _estimate_min_char_width(binary_img, W)
    min_dist   = max(min_char_w, (t1 - t0) // (n_chars * 2))

    # Find ALL local minima (no prominence filter here — we just want candidates)
    inv = proj_s[t0:t1].max() - proj_s[t0:t1]
    all_peaks, _ = find_peaks(inv, distance=min_dist, prominence=0)

    # Split into floor-passing (genuine gaps) and shallow (intra-char)
    genuine = [pk + t0 for pk in all_peaks if proj_s[pk + t0] <= gap_floor]
    shallow = [pk + t0 for pk in all_peaks if proj_s[pk + t0] > gap_floor]

    # Sort genuine by depth (deepest first = most confident gap)
    genuine.sort(key=lambda x: proj_s[x])

    if len(genuine) >= n_cuts:
        cuts = sorted(genuine[:n_cuts])
    else:
        # Use all genuine gaps, then fill remaining cuts from shallow,
        # sorted by projection value (lowest shallow = most gap-like)
        shallow.sort(key=lambda x: proj_s[x])
        combined = genuine + shallow
        cuts = sorted(combined[:n_cuts]) if len(combined) >= n_cuts else combined

        # Still not enough? Fill with equal-spaced minima
        while len(cuts) < n_cuts:
            spacing = (t1 - t0) / (len(cuts) + 2)
            for k in range(1, len(cuts) + 2):
                center = t0 + int(k * spacing)
                window = max(4, int(spacing * 0.35))
                lo = max(t0, center - window)
                hi = min(t1, center + window)
                if hi > lo:
                    local_min = lo + int(np.argmin(proj_s[lo:hi]))
                    if local_min not in cuts:
                        cuts.append(local_min)
                        break
            else:
                break  # no new cut found
            cuts.sort()

    boundaries = sorted(set([t0] + cuts[:n_cuts] + [t1]))
    return boundaries


# ======================================================================
# 7b.  BOUNDARY QUALITY FILTER  (remove cuts inside characters)
# ======================================================================

def filter_weak_boundaries(boundaries, proj_s, gap_floor_ratio=0.45):
    """
    Remove interior boundary positions where the projection profile does NOT
    show a genuine inter-character gap.

    Root cause this fixes
    ---------------------
    place_boundaries selects cuts from valley positions in the projection.
    But when count_characters over-estimates (e.g. returns 10 for 8 chars),
    some of the extra cuts land INSIDE characters because:
      • Characters with internal structure (hollow box, L-shape) have
        projection dips inside them.
      • The estimator mistake these dips for inter-character gaps.

    The fix: for each proposed cut at column x, check whether the projection
    NEAR x is actually low enough to constitute a real gap.  Two criteria
    must BOTH be true:

    1. LOCAL criterion  : proj near x < 55% of the LOCAL max of the two
                          adjacent segments.  Chars with internal dips still
                          have high proj on both sides; genuine gaps do not.

    2. GLOBAL criterion : proj near x ≤ gap_floor_ratio × global_max.
                          Intra-char dips stay above this floor; real gaps
                          typically fall below it.

    If either criterion fails → the cut is inside a character → REMOVE IT.
    The two adjacent segments will remain merged as one character.

    Parameters
    ----------
    boundaries      : list of x-positions [t0, cut1, cut2, ..., t1]
    proj_s          : smoothed vertical projection profile array
    gap_floor_ratio : same threshold used by other stages (default 0.45)

    Returns
    -------
    Filtered boundary list (shorter if weak cuts removed).
    """
    if len(boundaries) <= 2:
        return boundaries

    global_floor = proj_s.max() * gap_floor_ratio
    # Narrow window: only look very close to the cut column
    # Wide windows risk "borrowing" depth from a nearby genuine gap
    win = max(2, int(len(proj_s) * 0.008))   # ~0.8% of image width

    valid = [boundaries[0]]
    for i in range(1, len(boundaries) - 1):
        b        = int(boundaries[i])
        left_b   = int(boundaries[i - 1])
        right_b  = int(boundaries[i + 1])

        # Local max = max projection in the two adjacent segments
        left_proj  = proj_s[left_b:b]
        right_proj = proj_s[b:right_b]
        local_max  = max(
            float(left_proj.max())  if len(left_proj)  > 0 else 0,
            float(right_proj.max()) if len(right_proj) > 0 else 0
        )

        # Min projection in a tight window around the cut
        lo = max(0, b - win)
        hi = min(len(proj_s), b + win + 1)
        min_near = float(proj_s[lo:hi].min())

        local_ok  = (local_max == 0) or (min_near < local_max * 0.55)
        global_ok = min_near <= global_floor

        if local_ok and global_ok:
            valid.append(b)
        else:
            reasons = []
            if not local_ok:
                reasons.append(
                    f"local {min_near:.0f}/{local_max:.0f} = "
                    f"{min_near/local_max*100:.0f}% >= 55%"
                )
            if not global_ok:
                reasons.append(
                    f"global {min_near:.0f} > floor {global_floor:.0f}"
                )
            print(f"    Cut REMOVED (inside char) at x={b}: "
                  f"{' | '.join(reasons)}")

    valid.append(boundaries[-1])
    removed = len(boundaries) - len(valid)
    if removed:
        print(f"    {removed} weak cut(s) removed → "
              f"{len(valid)-1} genuine cuts remain")
    return valid


# ======================================================================
# 8.  SEGMENT VALIDATION + MZS SPLIT
# ======================================================================

def _mz_score(widths):
    w   = np.array(widths, dtype=float)
    M   = np.median(w)
    MAD = np.median(np.abs(w - M))
    MAD = MAD if MAD > 1e-6 else 1e-6
    return 0.675 * np.abs(w - M) / MAD, M, MAD


def validate_and_split(boundaries, proj_s, image_W, binary_img,
                       mzs_thresh=3.0):
    """
    Build segments from boundary list, then:
      1. Merge any overlapping segments (overlapping boxes = cut inside char)
      2. Iteratively MZS-split anomalously wide segments
    Returns list of cluster dicts.
    """
    H, W   = binary_img.shape[:2]
    min_w  = max(4, W // 60)

    def make_cluster(x0, x1, idx):
        strip = binary_img[:, x0:x1]
        rows  = np.where(np.any(strip == 255, axis=1))[0]
        if len(rows) == 0:
            return None
        return {"label": idx,
                "x": x0, "y": int(rows[0]),
                "w": x1 - x0, "h": int(rows[-1]) - int(rows[0]),
                "area": int(np.sum(strip == 255))}

    def best_split(x0, x1):
        sub = proj_s[x0:x1]
        if len(sub) < 4:
            return (x0 + x1) // 2
        return x0 + int(np.argmin(sub))

    segs = [(int(boundaries[i]), int(boundaries[i+1]))
            for i in range(len(boundaries)-1)
            if boundaries[i+1] - boundaries[i] >= min_w]

    # ── Overlap removal ──────────────────────────────────────────────────
    # Overlapping segments mean a cut landed inside a character.
    # Merge them back: if seg[i].x1 > seg[i+1].x0, combine into one.
    def merge_overlaps(segs):
        if not segs:
            return segs
        merged = [segs[0]]
        for x0, x1 in segs[1:]:
            px0, px1 = merged[-1]
            if x0 < px1:   # overlap: this segment starts before previous ends
                merged[-1] = (px0, max(px1, x1))
                print(f"    Overlap merged: [{px0},{px1}] + [{x0},{x1}] "
                      f"-> [{px0},{max(px1,x1)}]")
            else:
                merged.append((x0, x1))
        return merged

    segs = merge_overlaps(sorted(segs))

    def has_genuine_gap_inside(x0, x1):
        """
        Return True only if there is evidence of a genuine inter-character gap
        INSIDE segment [x0, x1].

        Two tests must BOTH pass:

        Test 1 – LOCAL dip:  the projection must dip to below 55% of the
          segment's OWN local maximum somewhere inside.  A wide single character
          has relatively uniform projection; two joined characters have a visible
          dip between them relative to their own peaks.

        Test 2 – ABSOLUTE floor:  the dip must also be below the global gap
          floor (GAP_FLOOR_RATIO * global_max).  This prevents splitting on
          a shallow dip that is still clearly within a character body.

        Using local max (not global) is critical: a wide character that is
        shorter than the tallest character in the image will have projection
        values that look low relative to the global max even though it is
        a single character with no internal gap.
        """
        sub = proj_s[x0:x1]
        if len(sub) < 4:
            return False

        local_max    = float(sub.max())
        local_min    = float(sub.min())
        global_floor = proj_s.max() * globals().get("GAP_FLOOR_RATIO", 0.45)

        # Test 1: dip below 55% of segment's own peak
        local_dip_ok = local_min < local_max * 0.55
        # Test 2: dip also below the global gap floor
        global_floor_ok = local_min <= global_floor

        ok = local_dip_ok and global_floor_ok
        print(f"      gap_check x={x0}-{x1}: local_min={local_min:.0f} "
              f"local_max={local_max:.0f} ({local_min/local_max*100:.0f}%) "
              f"global_floor={global_floor:.0f}  "
              f"split={'YES' if ok else 'NO (wide single char)'}")
        return ok

    # ── MZS split ────────────────────────────────────────────────────────
    changed = True
    passes  = 0
    while changed and passes < 3:
        changed = False
        passes += 1

        clusters = [make_cluster(x0, x1, i)
                    for i, (x0, x1) in enumerate(segs)]
        clusters = [c for c in clusters if c is not None]
        if len(clusters) < 3:
            break

        mzs, M, MAD = _mz_score([c["w"] for c in clusters])
        new_segs    = []
        for (x0, x1), z in zip(segs, mzs):
            if z > mzs_thresh:
                if not has_genuine_gap_inside(x0, x1):
                    # Wide but no internal gap → genuine wide single character
                    print(f"    MZS skip (no gap inside): x={x0} w={x1-x0} "
                          f"z={z:.2f}  min_proj="
                          f"{proj_s[x0:x1].min():.0f} > floor="
                          f"{proj_s.max()*globals().get('GAP_FLOOR_RATIO',0.45):.0f}")
                    new_segs.append((x0, x1))
                    continue
                mid = best_split(x0, x1)
                if mid - x0 >= min_w and x1 - mid >= min_w:
                    new_segs.extend([(x0, mid), (mid, x1)])
                    changed = True
                    print(f"    MZS split: x={x0} w={x1-x0} z={z:.2f}")
                    continue
            new_segs.append((x0, x1))
        segs = new_segs

    return [c for c in (make_cluster(x0, x1, i)
                        for i, (x0, x1) in enumerate(segs))
            if c is not None]


def _boundary_has_char_pixels(c, nxt, binary_img, min_density=0.08):
    """
    Inspect the actual binary image pixels at the cut column between c and nxt.

    Returns True  → cut column contains character pixels = wrong cut → MERGE
    Returns False → cut column is empty = genuine inter-character gap → KEEP SEPARATE

    This is the ground-truth check that projection profiles cannot provide
    for sparse inscriptions where projection dips inside characters.
    """
    H   = binary_img.shape[0]
    bx  = c["x"] + c["w"]
    win = max(1, int(binary_img.shape[1] * 0.005))   # ≈ 0.5% of width
    x0  = max(0, bx - win)
    x1  = min(binary_img.shape[1], bx + win + 1)
    region  = binary_img[:, x0:x1]
    density = np.sum(region == 255) / (H * max(1, x1 - x0))
    return density > min_density


def post_merge_narrow_segments(clusters, binary_img, proj_s,
                               min_width_ratio=0.55):
    """
    Merge adjacent segments only when two conditions are BOTH true:

    Condition 1 – Width (sliver check):
      At least one of the two segments is narrower than
      min_width_ratio × original_median_width.
      Three sub-cases: both narrow / left sliver / right sliver.

    Condition 2 – Boundary pixel gate (THE KEY FIX):
      The actual binary image at the cut column between the two segments
      must contain significant character pixels (density > 8% of col height).

      WHY: If the cut column is EMPTY → it is a genuine inter-character gap.
           Do NOT merge even if one segment looks narrow (it IS a small char).
           If the cut column has PIXELS → the cut went through a character.
           MERGE the two halves back into one character.

      This binary-image check is the ground truth that projection-profile
      analysis cannot provide for sparse inscriptions:
        • Projection dips inside characters → projection says "merge"
        • But the binary pixels at the gap column = 0 → binary says "keep"
        • Binary always wins.

    Exception: extreme slivers (< 30% of median) are merged regardless,
    because no complete Brahmi character can be that narrow.

    Stable median: computed once from original widths; never recomputed,
    preventing threshold creep across passes.
    """
    if len(clusters) < 2:
        return clusters

    H, W = binary_img.shape[:2]

    def make_cluster(x0, x1, idx):
        strip = binary_img[:, x0:x1]
        rows  = np.where(np.any(strip == 255, axis=1))[0]
        if len(rows) == 0:
            return None
        return {"label": idx,
                "x": x0, "y": int(rows[0]),
                "w": x1 - x0, "h": int(rows[-1]) - int(rows[0]),
                "area": int(np.sum(strip == 255))}

    base_median_w = float(np.median([c["w"] for c in clusters]))
    min_w         = min_width_ratio * base_median_w
    extreme_w     = 0.30 * base_median_w

    changed = True
    passes  = 0
    while changed and passes < 6:
        changed = False
        passes += 1

        new_clusters = []
        i = 0
        while i < len(clusters):
            c = clusters[i]

            # ── Tail segment ─────────────────────────────────────────
            if i == len(clusters) - 1:
                if new_clusters and c["w"] < min_w:
                    prev    = new_clusters[-1]
                    extreme = c["w"] < extreme_w
                    has_px  = _boundary_has_char_pixels(prev, c, binary_img)
                    if extreme or has_px:
                        x0 = prev["x"];  x1 = c["x"] + c["w"]
                        merged = make_cluster(x0, x1, len(new_clusters) - 1)
                        if merged:
                            new_clusters[-1] = merged
                            changed = True
                            gate = "extreme" if extreme else "wrong-cut pixels"
                            print(f"    Post-merge tail ({gate}): "
                                  f"w={prev['w']}+{c['w']}->{x1-x0}")
                        else:
                            new_clusters.append(c)
                    else:
                        print(f"    Post-merge SKIPPED (tail, genuine gap): "
                              f"w={prev['w']}+{c['w']}  boundary is empty")
                        new_clusters.append(c)
                else:
                    new_clusters.append(c)
                i += 1
                continue

            nxt = clusters[i + 1]
            w_c = c["w"];  w_n = nxt["w"]

            both_narrow  = (w_c < min_w) and (w_n < min_w)
            left_sliver  = (w_c < min_w)
            right_sliver = (w_n < min_w)

            if both_narrow or left_sliver or right_sliver:
                extreme = (w_c < extreme_w) or (w_n < extreme_w)
                has_px  = _boundary_has_char_pixels(c, nxt, binary_img)

                if extreme or has_px:
                    x0 = c["x"];  x1 = nxt["x"] + nxt["w"]
                    merged = make_cluster(x0, x1, len(new_clusters))
                    if merged:
                        new_clusters.append(merged)
                        i += 2
                        changed = True
                        reason = ("both narrow" if both_narrow else
                                  "left sliver"  if left_sliver  else
                                  "right sliver")
                        gate   = "extreme" if extreme else "wrong-cut pixels"
                        print(f"    Post-merge ({reason}, {gate}): "
                              f"w={w_c}+{w_n}->{x1-x0}  "
                              f"(median={base_median_w:.0f}  "
                              f"thresh={min_w:.1f})")
                        continue
                else:
                    print(f"    Post-merge SKIPPED (genuine gap): "
                          f"w={w_c}+{w_n}  "
                          f"boundary col is empty → two separate chars")

            new_clusters.append(c)
            i += 1

        for idx, c in enumerate(new_clusters):
            c["label"] = idx + 1
        clusters = new_clusters

    return clusters




def crop_characters(binary_img, clusters, padding=4):
    H, W  = binary_img.shape[:2]
    chars = []
    for i, c in enumerate(clusters):
        x0 = max(0, c["x"] - padding)
        y0 = max(0, c["y"] - padding)
        x1 = min(W, c["x"] + c["w"] + padding)
        y1 = min(H, c["y"] + c["h"] + padding)
        chars.append((i + 1, binary_img[y0:y1, x0:x1], c))
    return chars


# ======================================================================
# VISUALISATION
# ======================================================================

def vis_baseline(binary_img, baseline_info, out_path):
    vis  = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]
    poly = baseline_info["poly"]
    flow = baseline_info["flow_type"]
    curv = baseline_info["curvature"]
    ang  = baseline_info["angle_deg"]

    # Baseline curve
    for x in range(W - 1):
        y0c = int(poly(x));   y1c = int(poly(x + 1))
        if 0 <= y0c < H and 0 <= y1c < H:
            cv2.line(vis, (x, y0c), (x+1, y1c), (0, 220, 80), 2)

    # Centroid dots
    for cx, cy in baseline_info["centroids"]:
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 80, 255), -1)

    label = (f"Flow: {flow.upper()}  "
             f"curvature={curv:.1f}px  tilt={ang:.1f}deg  "
             f"degree={baseline_info['degree']}")
    cv2.rectangle(vis, (0, 0), (W, 24), (20, 20, 20), -1)
    cv2.putText(vis, label, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 230, 0), 1)
    cv2.imwrite(out_path, vis)
    print(f"  Baseline vis -> {out_path}")


def vis_count_signals(proj_s, detail, final_count, conf, image_W, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                   gridspec_kw={"height_ratios": [3, 1]})

    xs = np.arange(len(proj_s))
    ax1.fill_between(xs, proj_s, alpha=0.20, color="steelblue")
    ax1.plot(xs, proj_s, color="steelblue", lw=1.5, label="Projection profile")

    # Gap floor line — anything above this is intra-character structure
    gap_floor = proj_s.max() * 0.40
    ax1.axhline(gap_floor, color="darkorange", lw=1.5, ls="-.",
                label=f"Gap floor (40% of max = {gap_floor:.0f}px)")

    # Mark accepted valleys (those that passed the floor check)
    for v in detail["valleys"]:
        ax1.axvline(v, color="crimson", lw=1.5, ls="--", alpha=0.85,
                    label="Accepted cut" if v == detail["valleys"][0] else "")
        ax1.plot(v, proj_s[v], "rv", ms=8)

    ax1.set_xlabel("Column x (pixel)"); ax1.set_ylabel("White pixel count")
    conf_color = {"high": "green", "medium": "darkorange", "low": "red"}
    ax1.set_title(
        f"Vertical Projection Profile   |   "
        f"Final count = {final_count}   Confidence = {conf.upper()}   "
        f"[Valleys below orange line = real gaps]",
        color=conf_color.get(conf, "black"), fontsize=11
    )
    ax1.legend(fontsize=9)

    lbls = ["Projection (A)", "CCA (B)", "Gap-runs (C)", "FINAL"]
    vals = [detail["proj"], detail["cca"], detail["gaps"], final_count]
    clrs = ["#4C9BE8", "#5CB85C", "#F0AD4E",
            "#D9534F" if conf == "low" else
            "#F0AD4E" if conf == "medium" else "#2ECC71"]
    bars = ax2.bar(lbls, vals, color=clrs, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                 str(val), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(vals) + 3)
    ax2.set_ylabel("Count"); ax2.set_title("Estimator Comparison")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Count-signal fig -> {out_path}")


def vis_segmentation(binary_img, clusters, final_count, conf,
                     baseline_info, out_path):
    vis  = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]
    poly = baseline_info["poly"]

    # Draw baseline
    for x in range(W - 1):
        y0c = int(poly(x));   y1c = int(poly(x + 1))
        if 0 <= y0c < H and 0 <= y1c < H:
            cv2.line(vis, (x, y0c), (x+1, y1c), (0, 200, 80), 1)

    # Character boxes
    box_col = {"high": (0,210,0), "medium": (0,165,255), "low": (0,0,220)}
    col     = box_col.get(conf, (0, 200, 0))
    for c in clusters:
        x0 = c["x"];  x1 = c["x"] + c["w"]
        y0 = max(0, c["y"] - 2);  y1 = min(H, c["y"] + c["h"] + 2)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, str(c["label"]), (x0, max(y0 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    banner = (f"Characters: {final_count}   "
              f"Flow: {baseline_info['flow_type'].upper()}   "
              f"Confidence: {conf.upper()}")
    cv2.rectangle(vis, (0, 0), (W, 26), (25, 25, 25), -1)
    cv2.putText(vis, banner, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.imwrite(out_path, vis)
    print(f"  Segmentation vis -> {out_path}")


def vis_chars_grid(chars, out_path, max_n=30):
    n    = min(len(chars), max_n)
    cols = min(n, 10)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*2.0))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")
    for idx, (num, crop, _) in enumerate(chars[:n]):
        axes[idx].imshow(crop, cmap="gray")
        axes[idx].set_title(f"#{num}", fontsize=8)
        axes[idx].axis("off")
    fig.suptitle(f"Segmented Characters  (total = {len(chars)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Char grid -> {out_path}")


def vis_pipeline(stages, out_path):
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Pipeline summary -> {out_path}")


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


def detect_text_rows(binary_img, min_row_height_frac=0.10):
    """
    Detect how many text rows exist using horizontal projection analysis.

    Characters in a multi-row inscription produce PEAKS in the row-wise
    white-pixel sum.  Valleys between peaks = row separators.

    Returns
    ───────
    row_bands : list of (y_start, y_end) tuples, one per detected text row.
                Each band includes the full vertical extent of that row.
    n_rows    : number of text rows found (1 or more)
    """
    H, W     = binary_img.shape[:2]
    h_proj   = np.sum(binary_img == 255, axis=1).astype(float)
    h_smooth = gaussian_filter1d(h_proj, sigma=4)

    # Threshold: rows with significant character content
    threshold    = max(h_smooth.max() * 0.12, 5.0)
    active       = (h_smooth >= threshold).astype(int)

    # Find contiguous active bands
    bands = []
    in_band, start = False, 0
    for y in range(H):
        if active[y] and not in_band:
            start, in_band = y, True
        elif not active[y] and in_band:
            bands.append((start, y))
            in_band = False
    if in_band:
        bands.append((start, H))

    # Merge bands that are very close together (< 8px gap)
    merged = []
    for band in bands:
        if merged and band[0] - merged[-1][1] < 8:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(list(band))

    # Filter out tiny bands (< min_row_height_frac * H)
    min_h = max(8, int(H * min_row_height_frac))
    merged = [(s, e) for s, e in merged if e - s >= min_h]

    n_rows = len(merged)
    print(f"  Text row detection: {n_rows} row(s) found — "
          f"{[(s, e, e-s) for s, e in merged]}")

    if not merged:
        return [(0, H)], 1   # fallback: full image is one row

    return merged, n_rows


def segment_one_row(binary_row, row_y_offset, gap_floor_ratio,
                    mzs_threshold, out_dir, row_idx):
    """
    Run the full segmentation pipeline on ONE text row strip.
    Returns list of cluster dicts with y-coordinates adjusted by row_y_offset.
    """
    if np.sum(binary_row == 255) < 50:
        return []

    # Baseline + rectification for this row
    baseline = detect_baseline(binary_row)
    rectified, _ = rectify(binary_row, baseline)

    # Count characters in this row
    count, conf, detail = count_characters(rectified, baseline)

    # Save per-row signals
    vis_path = os.path.join(out_dir, f"row{row_idx:02d}_signals.png")
    vis_count_signals(detail["proj_s"], detail, count, conf,
                      rectified.shape[1], vis_path)

    # Place boundaries and validate
    boundaries = place_boundaries(
        detail["proj_s"], count, rectified.shape[1], rectified)
    boundaries = filter_weak_boundaries(
        boundaries, detail["proj_s"], gap_floor_ratio)
    clusters = validate_and_split(
        boundaries, detail["proj_s"],
        rectified.shape[1], rectified, mzs_thresh=mzs_threshold)
    clusters = post_merge_narrow_segments(
        clusters, rectified, detail["proj_s"])

    # Crop characters for this row
    chars = crop_characters(rectified, clusters)

    print(f"  Row {row_idx}: count={len(clusters)}  conf={conf}  "
          f"flow={baseline['flow_type']}")

    # Adjust cluster y-coordinates back to full-image space
    for c in clusters:
        c["y"] += row_y_offset

    return clusters, chars, baseline, detail



    """
    Automatically measure image characteristics and return the optimal
    processing parameters for that specific image.

    This is the standardisation solution: instead of every image needing
    manual parameter tuning, this function analyses the image and sets
    parameters adaptively so all images get accurate output.

    Measurements
    ─────────────
    noise_ratio   : mean(|Laplacian|) / dynamic_range
                    Measures texture noise relative to the image's contrast.
                    Low  (< 0.05) = clean, high-contrast scan.
                    High (> 0.15) = very noisy, degraded stone texture.

    contrast      : 95th percentile - 5th percentile of pixel values.
                    High contrast = clear ink vs background separation.
                    Low contrast  = faded/worn inscription.

    fg_ratio      : fraction of pixels above Otsu threshold (foreground %).
                    Very low (< 0.10) = sparse text, few connected chars.
                    High     (> 0.40) = dense/merged text blobs.

    blob_ratio    : area of largest CCA blob / total image area.
                    High blob_ratio = characters are joined through noise.

    Parameter mapping
    ─────────────────
    nlm_h          → stronger for noisier images (noise_ratio)
    bilateral      → always True (improves edge preservation)
    otsu_bias      → higher for low-contrast images
    white_thresh   → lower for images with fine noise speckles
    black_thresh   → lower for sparse images (preserve thin strokes)
    gap_floor      → lower for noisy/joined text; higher for clean text
    """
    raw  = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw

    H, W = gray.shape[:2]

    # ── Noise measurement ────────────────────────────────────────────────
    lap         = cv2.Laplacian(gray.astype(float), cv2.CV_64F)
    dyn_range   = max(1.0, float(gray.max()) - float(gray.min()))
    noise_ratio = float(np.abs(lap).mean()) / dyn_range

    # ── Contrast ─────────────────────────────────────────────────────────
    contrast    = float(np.percentile(gray, 95) - np.percentile(gray, 5))

    # ── Foreground density (quick Otsu) ──────────────────────────────────
    _, bw      = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio   = float(np.mean(bw == 255))
    if fg_ratio > 0.55:
        fg_ratio = 1.0 - fg_ratio   # image was inverted; report minority

    # ── Blob connectivity (with light denoising) ─────────────────────────
    den_quick  = cv2.fastNlMeansDenoising(gray, None, h=15,
                                          templateWindowSize=7,
                                          searchWindowSize=21)
    _, bw2     = cv2.threshold(den_quick, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw2 == 255) > 0.55:
        bw2 = cv2.bitwise_not(bw2)
    n2, _, st2, _ = cv2.connectedComponentsWithStats(bw2, connectivity=8)
    areas2     = [int(st2[i, cv2.CC_STAT_AREA]) for i in range(1, n2)]
    blob_ratio = float(max(areas2)) / (H * W) if areas2 else 0.0

    print(f"  Auto-calibrate:  noise_ratio={noise_ratio:.4f}  "
          f"contrast={contrast:.0f}  fg_ratio={fg_ratio:.3f}  "
          f"blob_ratio={blob_ratio:.4f}")

    # ── Parameter mapping ────────────────────────────────────────────────

    # nlm_h: 5 (clean) → 30 (very noisy)
    if noise_ratio < 0.05:
        nlm_h = 5
    elif noise_ratio < 0.08:
        nlm_h = 10
    elif noise_ratio < 0.12:
        nlm_h = 18
    elif noise_ratio < 0.18:
        nlm_h = 25
    else:
        nlm_h = 30

    # otsu_bias: 20 normal; 30 for low contrast (capture more ink)
    otsu_bias = 30 if contrast < 150 else 20

    # white_thresh / black_thresh:
    # Noisy images have many tiny speckles → lower threshold to kill more
    # Clean images have fewer, larger blobs → higher threshold to be safe
    if noise_ratio < 0.06:
        noise_thresh = 50
    elif noise_ratio < 0.10:
        noise_thresh = 100
    elif noise_ratio < 0.15:
        noise_thresh = 150
    else:
        noise_thresh = 200

    # gap_floor:
    # Noisy / heavily joined images → lower floor (accept shallow valleys)
    # Clean images with real gaps    → higher floor (reject intra-char dips)
    if blob_ratio > 0.15:
        gap_floor = 0.30          # severely connected
    elif blob_ratio > 0.08:
        gap_floor = 0.35          # moderately connected
    elif noise_ratio > 0.12:
        gap_floor = 0.40          # noisy but separable
    elif noise_ratio > 0.06:
        gap_floor = 0.45          # normal
    else:
        gap_floor = 0.55          # clean, well-separated characters

    params = {
        "nlm_h":              nlm_h,
        "bilateral":          True,
        "otsu_bias":          otsu_bias,
        "dilate_kernel":      (3, 3),
        "dilate_iters":       1,
        "white_noise_thresh": noise_thresh,
        "black_noise_thresh": noise_thresh,
        "gap_floor_ratio":    gap_floor,
        "mzs_threshold":      3.0,
        # diagnostics (not passed to run_module1)
        "_noise_ratio":       round(noise_ratio, 4),
        "_contrast":          round(contrast, 1),
        "_fg_ratio":          round(fg_ratio, 3),
        "_blob_ratio":        round(blob_ratio, 4),
    }

    print(f"  → nlm_h={nlm_h}  noise_thresh={noise_thresh}  "
          f"gap_floor={gap_floor}  otsu_bias={otsu_bias}")
    return params


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_module1(image_path: str,
                out_dir: str             = "output_module1",
                white_noise_thresh: int  = None,
                black_noise_thresh: int  = None,
                mzs_threshold: float     = 3.0,
                gap_floor_ratio: float   = None,
                nlm_h: int               = None,
                bilateral: bool          = True,
                otsu_bias: int           = None,
                dilate_kernel: tuple     = (3, 3),
                dilate_iters: int        = 1,
                auto_params: bool        = True,
                save_individual_chars: bool = True,
                show_plots: bool         = False) -> dict:
    """
    Full Module 1 pipeline.

    auto_params : bool (default True)
        When True, auto_calibrate() measures the image and sets all
        processing parameters automatically.  Any parameter you pass
        explicitly OVERRIDES the auto-detected value.  Set to False
        only if you want full manual control of all parameters.

    Preprocessing controls (override auto_params when specified)
    ────────────────────────────────────────────────────────────
    nlm_h          : Non-Local Means filter strength.
    bilateral      : Bilateral edge-preserving filter after NLMeans.
    otsu_bias      : Subtracted from Otsu threshold before binarising.
    dilate_kernel  : Morphological closing kernel (w, h).
    dilate_iters   : Closing iterations.

    Segmentation controls
    ─────────────────────
    gap_floor_ratio : Valley depth threshold.
                      RAISE if too many chars, LOWER if too few.
    mzs_threshold   : Modified Z-Score split threshold.
    """
    import sys as _sys

    # ── Auto-calibrate first, then apply any explicit overrides ──────────
    if auto_params:
        print(f"\n[AUTO] Calibrating parameters for: {image_path}")
        cal = auto_calibrate(image_path)
        # Use calibrated value unless caller passed an explicit override
        _nlm_h        = nlm_h              if nlm_h              is not None else cal["nlm_h"]
        _bilateral    = bilateral
        _otsu_bias    = otsu_bias          if otsu_bias           is not None else cal["otsu_bias"]
        _white_thresh = white_noise_thresh if white_noise_thresh  is not None else cal["white_noise_thresh"]
        _black_thresh = black_noise_thresh if black_noise_thresh  is not None else cal["black_noise_thresh"]
        _gap_floor    = gap_floor_ratio    if gap_floor_ratio     is not None else cal["gap_floor_ratio"]
    else:
        # Full manual mode — use defaults if not specified
        _nlm_h        = nlm_h              if nlm_h              is not None else 10
        _bilateral    = bilateral
        _otsu_bias    = otsu_bias          if otsu_bias           is not None else 20
        _white_thresh = white_noise_thresh if white_noise_thresh  is not None else 200
        _black_thresh = black_noise_thresh if black_noise_thresh  is not None else 200
        _gap_floor    = gap_floor_ratio    if gap_floor_ratio     is not None else 0.45

    _sys.modules[__name__].GAP_FLOOR_RATIO = _gap_floor
    os.makedirs(out_dir, exist_ok=True)
    chars_dir = os.path.join(out_dir, "characters")
    os.makedirs(chars_dir, exist_ok=True)
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  Module 1  Brahmi Inscription Segmentation")
    print(f"  Input  : {image_path}")
    print(f"  Output : {out_dir}")
    print(f"  Params : nlm_h={_nlm_h}  bilateral={_bilateral}  "
          f"otsu_bias={_otsu_bias}")
    print(f"           noise_thresh={_white_thresh}/{_black_thresh}  "
          f"gap_floor={_gap_floor}  auto={'yes' if auto_params else 'no'}")
    print(sep)

    # Load
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    print(f"\n[0] Loaded  shape={raw.shape}")
    cv2.imwrite(os.path.join(out_dir, "00_raw.png"), raw)

    # Step 1 — Preprocessing (now with quality metrics)
    print("\n[1] Preprocessing ...")
    gray, binary_clean, dilated_crop, metrics = preprocess(
        raw,
        nlm_h        = _nlm_h,
        bilateral    = _bilateral,
        otsu_bias    = _otsu_bias,
        dilate_kernel= dilate_kernel,
        dilate_iters = dilate_iters,
    )
    cv2.imwrite(os.path.join(out_dir, "01a_gray.png"),         gray)
    cv2.imwrite(os.path.join(out_dir, "01b_binary_clean.png"), binary_clean)
    cv2.imwrite(os.path.join(out_dir, "01c_dilated_crop.png"), dilated_crop)

    # Save quality metrics as a text report
    metrics_path = os.path.join(out_dir, "01_quality_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Preprocessing Quality Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Image        : {image_path}\n")
        f.write(f"NLM h        : {nlm_h}\n")
        f.write(f"Bilateral    : {bilateral}\n")
        f.write(f"Otsu bias    : {otsu_bias}\n")
        f.write(f"Dilate kernel: {dilate_kernel}  iters={dilate_iters}\n")
        f.write("-" * 40 + "\n")
        f.write(f"PSNR              : {metrics['psnr']} dB\n")
        f.write(f"  (Higher = more signal preserved after denoising)\n")
        f.write(f"SSIM              : {metrics['ssim']}\n")
        f.write(f"  (1.0 = perfect structural similarity to original)\n")
        f.write(f"Laplacian variance: {metrics['laplacian_var']}\n")
        f.write(f"  (Higher = sharper binarised image, better edges)\n")
        f.write(f"Edge retention    : {metrics['edge_retention']*100:.1f}%\n")
        f.write(f"  (Fraction of original character edges preserved)\n")
        f.write("-" * 40 + "\n")
        f.write("Reference paper benchmarks (Ch.7.1.1):\n")
        f.write("  Avg success rate on 100 estampages: 95%+\n")
        f.write("  F1 on 6 sample images: >90% all, 100% on 4/6\n")
    print(f"  Quality metrics saved -> {metrics_path}")

    # Step 2
    print("\n[2] Crop to inscription region ...")
    cropped, bbox = crop_to_inscription(dilated_crop, binary_clean)
    cv2.imwrite(os.path.join(out_dir, "02_cropped.png"), cropped)
    print(f"  Crop bbox: {bbox}")

    # Step 3
    print(f"\n[3] Noise removal (white={_white_thresh}, "
          f"black={_black_thresh}) ...")
    p1, inv, denoised = noise_removal(cropped, _white_thresh, _black_thresh)
    cv2.imwrite(os.path.join(out_dir, "03a_pass1.png"),    p1)
    cv2.imwrite(os.path.join(out_dir, "03b_inverted.png"), inv)
    cv2.imwrite(os.path.join(out_dir, "03c_denoised.png"), denoised)

    # Step 3c — Remove border-touching blobs (white paper/frame artefacts)
    print("\n[3c] Removing border-touching noise blobs ...")
    denoised = remove_border_blobs(denoised)
    cv2.imwrite(os.path.join(out_dir, "03c2_no_border.png"), denoised)

    # Step 3b — Character band extraction (spline-guided permanent noise removal)
    print("\n[3b] Character band extraction ...")
    denoised, baseline_rough, band_half = extract_character_band(
        denoised,
        padding=12,
        band_height_factor=1.3,
        save_vis_path=os.path.join(out_dir, "03d_band_vis.png"),
    )
    cv2.imwrite(os.path.join(out_dir, "03d_band_masked.png"), denoised)

    # Step 3d — Detect number of text rows
    print("\n[3d] Detecting text row structure ...")
    row_bands, n_rows = detect_text_rows(denoised)
    print(f"  Found {n_rows} text row(s)")

    # ── MULTI-ROW PATH ────────────────────────────────────────────────────
    if n_rows > 1:
        print(f"\n[MULTI-ROW] Processing {n_rows} rows independently ...")
        all_clusters = []
        all_chars    = []
        baselines    = []

        for ri, (y0, y1) in enumerate(row_bands):
            pad      = 5
            strip_y0 = max(0, y0 - pad)
            strip_y1 = min(denoised.shape[0], y1 + pad)
            row_strip = denoised[strip_y0:strip_y1, :]
            cv2.imwrite(os.path.join(out_dir,
                        f"row{ri:02d}_strip.png"), row_strip)

            result = segment_one_row(
                row_strip, strip_y0, _gap_floor,
                mzs_threshold, out_dir, ri)

            if result:
                row_clusters, row_chars, row_base, row_detail = result
                # Re-label globally
                label_offset = len(all_clusters)
                for c in row_clusters:
                    c["label"] += label_offset
                all_clusters.extend(row_clusters)
                all_chars.extend(row_chars)
                baselines.append(row_base)

        clusters = all_clusters
        chars    = all_chars
        # Use first row's baseline for visualisation
        baseline = baselines[0] if baselines else baseline_rough

        # Draw combined segmentation visualisation
        vis_combined = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        conf = "medium"
        for c in clusters:
            x0, y0c = c["x"], max(0, c["y"] - 2)
            x1, y1c = c["x"] + c["w"], min(denoised.shape[0], c["y"] + c["h"] + 2)
            cv2.rectangle(vis_combined, (x0, y0c), (x1, y1c), (0, 165, 255), 2)
            cv2.putText(vis_combined, str(c["label"]),
                        (x0, max(y0c - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1)
        banner = (f"Characters: {len(clusters)}   "
                  f"Rows: {n_rows}   Flow: MULTI-ROW")
        cv2.rectangle(vis_combined, (0, 0), (denoised.shape[1], 26), (25, 25, 25), -1)
        cv2.putText(vis_combined, banner, (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, "08_segmentation.png"), vis_combined)
        vis_chars_grid(chars, os.path.join(out_dir, "09_chars_grid.png"))

        detail = {"proj": 0, "cca": 0, "gaps": 0,
                  "valleys": [], "proj_s": np.zeros(10)}

    else:
        # ── SINGLE-ROW PATH (original pipeline) ──────────────────────────
        # Step 4 — Re-detect baseline on the now-clean masked image
        print("\n[4] Baseline detection on cleaned image ...")
        baseline = detect_baseline(denoised)
        vis_baseline(denoised, baseline,
                     os.path.join(out_dir, "04_baseline.png"))

        # Step 5
        print(f"\n[5] Rectification (flow={baseline['flow_type']}) ...")
        rectified, col_offsets = rectify(denoised, baseline)
        cv2.imwrite(os.path.join(out_dir, "05_rectified.png"), rectified)
        if baseline["flow_type"] == "straight":
            print("  Straight baseline -> no geometric correction applied")
        else:
            print(f"  Curved -> max column shift = {np.abs(col_offsets).max()}px")

        # Step 6
        print("\n[6] Multi-signal character counting ...")
        count, conf, detail = count_characters(rectified, baseline)
        vis_count_signals(
            detail["proj_s"], detail, count, conf,
            rectified.shape[1],
            os.path.join(out_dir, "06_count_signals.png")
        )

        # Step 7
        print(f"\n[7] Placing {count-1} boundaries for {count} characters ...")
        boundaries = place_boundaries(
            detail["proj_s"], count, rectified.shape[1], rectified)
        print(f"  Raw boundaries ({len(boundaries)-1} cuts): {boundaries}")

        # Step 7b
        print(f"\n[7b] Filtering weak boundaries ...")
        boundaries = filter_weak_boundaries(
            boundaries, detail["proj_s"], _gap_floor)
        print(f"  After filter ({len(boundaries)-1} cuts): {boundaries}")

        # Step 8
        print("\n[8] Segment validation and MZS split ...")
        clusters = validate_and_split(
            boundaries, detail["proj_s"],
            rectified.shape[1], rectified,
            mzs_thresh=mzs_threshold
        )
        print(f"  After MZS split: {len(clusters)} segments")

        # Step 8b
        print("\n[8b] Post-merge: collapsing over-split narrow segments ...")
        clusters = post_merge_narrow_segments(
            clusters, rectified, detail["proj_s"]
        )
        print(f"  Final segment count: {len(clusters)}")

        vis_segmentation(rectified, clusters, len(clusters), conf,
                         baseline, os.path.join(out_dir, "08_segmentation.png"))

        # Step 9
        print("\n[9] Cropping individual characters ...")
        chars = crop_characters(rectified, clusters)
        vis_chars_grid(chars, os.path.join(out_dir, "09_chars_grid.png"))

    # ── Save individual chars ─────────────────────────────────────────────
    if save_individual_chars:
        for num, crop_img, _ in chars:
            cv2.imwrite(os.path.join(chars_dir, f"char_{num:03d}.png"), crop_img)
        print(f"  Saved {len(chars)} chars -> {chars_dir}/")

    vis_pipeline([
        ("Gray",         gray),
        ("Binary",       binary_clean),
        ("No border",    denoised),
    ], os.path.join(out_dir, "pipeline_summary.png"))

    print(f"\n{'─'*64}")
    print(f"  CHARACTER COUNT  : {len(clusters)}")
    print(f"  Estimator votes  : proj={detail['proj']}  "
          f"cca={detail['cca']}  gaps={detail['gaps']}")
    print(f"  Confidence       : {conf.upper()}")
    print(f"  Flow type        : {baseline['flow_type'].upper()}")
    print(f"  Curvature        : {baseline['curvature']:.1f} px")
    print(f"  Tilt angle       : {baseline['angle_deg']:.1f} deg")
    print(f"  Gap floor used   : {_gap_floor*100:.0f}% of max projection "
          f"({'auto' if auto_params else 'manual'})")
    print(f"  NLM h used       : {_nlm_h}  "
          f"noise_thresh={_white_thresh}/{_black_thresh}")
    print(f"{'─'*64}")
    print(f"  Preprocessing quality (ref paper metrics):")
    print(f"    PSNR            : {metrics['psnr']} dB")
    print(f"    SSIM            : {metrics['ssim']}")
    print(f"    Laplacian var   : {metrics['laplacian_var']:.0f}")
    print(f"    Edge retention  : {metrics['edge_retention']*100:.1f}%")
    print(f"{'─'*64}")
    print(f"  TIP: auto_params=True (default) → parameters auto-tuned per image")
    print(f"  TIP: override any param via CLI flags, e.g. --gap_floor 0.55")
    print(f"  TIP: use --no_auto to disable auto-calibration (full manual)")
    print(f"  Output dir       : {out_dir}")
    print(f"{'─'*64}\n")

    return {
        "chars":             chars,
        "count":             len(clusters),
        "confidence":        conf,
        "flow_type":         baseline["flow_type"],
        "baseline_info":     baseline,
        "clusters":          clusters,
        "proj_detail":       detail,
        "quality_metrics":   metrics,
        "params_used":       {
            "nlm_h":              _nlm_h,
            "bilateral":          _bilateral,
            "otsu_bias":          _otsu_bias,
            "white_noise_thresh": _white_thresh,
            "black_noise_thresh": _black_thresh,
            "gap_floor_ratio":    _gap_floor,
            "auto_params":        auto_params,
        },
    }


# ======================================================================
# BATCH
# ======================================================================

def batch_process(input_dir, out_root="batch_output", **kwargs):
    exts   = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted(p for p in Path(input_dir).iterdir()
                    if p.suffix.lower() in exts)
    if not images:
        print(f"No images in {input_dir}"); return

    summary = []
    for img_path in images:
        od = os.path.join(out_root, img_path.stem)
        try:
            r = run_module1(str(img_path), out_dir=od, **kwargs)
            m = r.get("quality_metrics", {})
            summary.append((img_path.name, r["count"],
                            r["confidence"], r["flow_type"],
                            m.get("psnr", 0), m.get("ssim", 0),
                            m.get("edge_retention", 0)))
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}")
            summary.append((img_path.name, -1, "error", "?", 0, 0, 0))

    print("\n" + "=" * 80)
    print(f"{'Image':<28} {'Count':>6}  {'Conf':<8} {'Flow':<9} "
          f"{'PSNR':>7}  {'SSIM':>6}  {'EdgeRet':>8}")
    print("-" * 80)
    for name, cnt, conf, flow, psnr, ssim, edg in summary:
        c = str(cnt) if cnt >= 0 else "FAIL"
        print(f"{name:<28} {c:>6}  {conf:<8} {flow:<9} "
              f"{psnr:>7.1f}  {ssim:>6.3f}  {edg*100:>7.1f}%")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Module 1 - Brahmi Inscription Preprocessing & Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input",
                    help="Image file, or directory (use --batch)")
    ap.add_argument("--out",          default="output_module1")

    # ── Preprocessing (new — from reference paper Ch.6) ─────────────────
    grp_pre = ap.add_argument_group("Preprocessing (reference paper additions)")
    grp_pre.add_argument("--nlm_h",         type=int,   default=10,
                         help="Non-Local Means filter strength. "
                              "Higher=more smoothing. "
                              "Try 20-30 for very noisy scans.")
    grp_pre.add_argument("--bilateral",     action="store_true", default=True,
                         help="Add bilateral edge-preserving filter after NLMeans "
                              "(default ON). Disable with --no_bilateral.")
    grp_pre.add_argument("--no_bilateral",  dest="bilateral", action="store_false",
                         help="Disable bilateral filter (already-clean images).")
    grp_pre.add_argument("--otsu_bias",     type=int,   default=20,
                         help="Bias subtracted from Otsu threshold. "
                              "Lower for faded inscriptions.")
    grp_pre.add_argument("--dilate_kernel", type=int,   nargs=2, default=[3, 3],
                         metavar=("W", "H"),
                         help="Morphological closing kernel size. "
                              "Use '5 5' for heavily fragmented strokes.")
    grp_pre.add_argument("--dilate_iters",  type=int,   default=1,
                         help="Morphological closing iterations. "
                              "Increase for low-DPI images.")

    # ── Noise removal ────────────────────────────────────────────────────
    grp_noise = ap.add_argument_group("Noise removal (CCA size filters)")
    grp_noise.add_argument("--white_thresh", type=int, default=200,
                            help="White cluster removal threshold (px²).")
    grp_noise.add_argument("--black_thresh", type=int, default=200,
                            help="Black cluster removal threshold (px²).")

    # ── Segmentation ─────────────────────────────────────────────────────
    grp_seg = ap.add_argument_group("Segmentation")
    grp_seg.add_argument("--mzs",       type=float, default=3.0,
                         help="Modified Z-Score threshold for wide-segment split.")
    grp_seg.add_argument("--gap_floor", type=float, default=0.45,
                         help="Gap-floor ratio 0-1. "
                              "RAISE if too many chars, LOWER if too few.")

    ap.add_argument("--show",    action="store_true", help="Show matplotlib plots.")
    ap.add_argument("--batch",   action="store_true", help="Process whole directory.")
    ap.add_argument("--no_auto", action="store_true",
                    help="Disable auto-calibration. Use explicit flags for all params.")
    args = ap.parse_args()

    kw = dict(
        white_noise_thresh = args.white_thresh  if args.no_auto else None,
        black_noise_thresh = args.black_thresh  if args.no_auto else None,
        mzs_threshold      = args.mzs,
        gap_floor_ratio    = args.gap_floor     if args.no_auto else None,
        nlm_h              = args.nlm_h         if args.no_auto else None,
        bilateral          = args.bilateral,
        otsu_bias          = args.otsu_bias     if args.no_auto else None,
        dilate_kernel      = tuple(args.dilate_kernel),
        dilate_iters       = args.dilate_iters,
        auto_params        = not args.no_auto,
        show_plots         = args.show,
    )

    if args.batch:
        batch_process(args.input, out_root=args.out, **kw)
    else:
        run_module1(args.input, out_dir=args.out, **kw)