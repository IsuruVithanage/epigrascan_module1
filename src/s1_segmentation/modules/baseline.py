import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as _ssim
import os
from pathlib import Path



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