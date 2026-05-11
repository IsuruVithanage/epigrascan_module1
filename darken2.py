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
import os
from pathlib import Path


# ======================================================================
# 1.  PREPROCESSING
# ======================================================================

def preprocess(raw_bgr):
    """
    Returns
    -------
    gray          : original grayscale
    binary_clean  : WHITE text on BLACK -- used for ALL character work
    dilated_crop  : aggressively dilated -- used ONLY to find crop bbox
    """
    gray = (cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
            if raw_bgr.ndim == 3 else raw_bgr.copy())

    den = cv2.fastNlMeansDenoising(gray, None, h=10,
                                   templateWindowSize=7, searchWindowSize=21)

    t_glob, _ = cv2.threshold(den, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_adj = max(0, t_glob - 20)
    _, binary = cv2.threshold(den, t_adj, 255, cv2.THRESH_BINARY)
    print(f"  Otsu T_global={t_glob:.0f}  T_adjusted={t_adj:.0f}")

    # Auto-polarity: ensure WHITE = foreground text
    if np.mean(binary == 255) > 0.55:
        binary = cv2.bitwise_not(binary)
        print("  Polarity: inverted (background was white)")
    else:
        print("  Polarity: kept (background already black)")

    # Gentle 3x3 closing: repairs tiny stroke gaps without bloat
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se3, iterations=1)

    # Large dilation for crop-region detection ONLY
    se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    dilated_crop = cv2.dilate(binary_clean, se_h, iterations=1)
    dilated_crop = cv2.dilate(dilated_crop, se_v, iterations=1)

    return gray, binary_clean, dilated_crop


# ======================================================================
# 2.  CROP TO INSCRIPTION REGION
# ======================================================================

def crop_to_inscription(dilated_crop, binary_clean, pad=8):
    """
    Locate largest white blob in dilated_crop -> bounding box.
    Crop that region from binary_clean (NOT from dilated).
    """
    nl, _, st, _ = cv2.connectedComponentsWithStats(dilated_crop, connectivity=8)
    if nl < 2:
        return binary_clean.copy(), (0, 0, binary_clean.shape[1], binary_clean.shape[0])

    lbl = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    x, y = st[lbl, cv2.CC_STAT_LEFT], st[lbl, cv2.CC_STAT_TOP]
    w, h = st[lbl, cv2.CC_STAT_WIDTH], st[lbl, cv2.CC_STAT_HEIGHT]

    H, W = binary_clean.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return binary_clean[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


# ======================================================================
# 3.  TWO-PASS NOISE REMOVAL
# ======================================================================

def _cca_filter(img, threshold, kill_color, replace_color):
    mask = img if kill_color == 255 else cv2.bitwise_not(img)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = img.copy()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < threshold:
            out[lbl == i] = replace_color
    return out


def noise_removal(cropped, white_thresh=200, black_thresh=200):
    p1  = _cca_filter(cropped, white_thresh, kill_color=255, replace_color=0)
    inv = cv2.bitwise_not(p1)
    p2  = _cca_filter(inv,    black_thresh, kill_color=255, replace_color=0)
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


def detect_baseline(binary_img):
    """
    Fit a polynomial to centroid y-coordinates vs x.

    Degree 1 -> linear (straight)
    Degree 2 -> quadratic (curved)
    Degree 3 -> cubic (wavy)

    Upgrade degree only when residual drops >= 30%.

    Returns dict with: flow_type, poly, degree, residual_std,
                       curvature, angle_deg, centroids, image_W, image_H
    """
    H, W = binary_img.shape[:2]
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

    fits = {}
    for deg in [1, 2, 3]:
        cf   = np.polyfit(xs, ys, deg)
        poly = np.poly1d(cf)
        resid = ys - poly(xs)
        fits[deg] = {"cf": cf, "poly": poly, "std": float(np.std(resid))}

    chosen = 1
    if fits[2]["std"] < fits[1]["std"] * 0.70:
        chosen = 2
    if fits[3]["std"] < fits[2]["std"] * 0.70:
        chosen = 3

    poly  = fits[chosen]["poly"]
    std   = fits[chosen]["std"]
    xs_f  = np.linspace(0, W, 300)
    ys_f  = poly(xs_f)
    curv  = float(ys_f.max() - ys_f.min())

    if chosen == 1 and std < 0.05 * H:
        flow = "straight"
    elif curv < 0.08 * H:
        flow = "straight"
    elif chosen == 2:
        flow = "curved"
    else:
        flow = "wavy"

    angle = float(np.degrees(np.arctan2(ys[-1]-ys[0], xs[-1]-xs[0]))) \
            if len(xs) >= 2 else 0.0

    print(f"  Baseline: flow={flow}  degree={chosen}  "
          f"curvature={curv:.1f}px  residual_std={std:.1f}px  tilt={angle:.1f}deg")

    return {
        "flow_type":    flow,
        "poly":         poly,
        "poly_coeffs":  fits[chosen]["cf"],
        "degree":       chosen,
        "residual_std": std,
        "curvature":    curv,
        "angle_deg":    angle,
        "centroids":    list(zip(xs.tolist(), ys.tolist())),
        "image_W": W, "image_H": H,
    }


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


def post_merge_narrow_segments(clusters, binary_img, proj_s,
                               min_width_ratio=0.55):
    """
    After splitting, merge any adjacent pair where BOTH segments are
    narrower than min_width_ratio * median_width.

    This is the critical over-split repair step.  When a projection valley
    is detected inside a character (hollow shape, thin bridge of noise),
    the two halves will both be narrower than a normal character.
    Merging them restores the correct boundary.

    Also merges a very narrow segment (< 40% of median) unconditionally
    with its smaller neighbour — these are almost always fragment slivers
    from the edge of a character, not real characters.

    Parameters
    ----------
    min_width_ratio : adjacent pair merged if BOTH < this * median_w
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

    changed = True
    passes  = 0
    while changed and passes < 5:
        changed = False
        passes += 1

        widths   = [c["w"] for c in clusters]
        median_w = float(np.median(widths))
        min_pair = min_width_ratio * median_w    # both must be below this
        min_solo = 0.40 * median_w               # unconditional merge threshold

        new_clusters = []
        i = 0
        while i < len(clusters):
            c = clusters[i]
            if i == len(clusters) - 1:
                new_clusters.append(c)
                i += 1
                continue

            nxt = clusters[i + 1]
            w_c = c["w"]
            w_n = nxt["w"]

            # Case 1: BOTH neighbours are too narrow → merge
            both_narrow = (w_c < min_pair) and (w_n < min_pair)
            # Case 2: current segment is a sliver (< 40% median) → merge with next
            solo_sliver = (w_c < min_solo)

            if both_narrow or solo_sliver:
                x0 = c["x"]
                x1 = nxt["x"] + nxt["w"]
                merged = make_cluster(x0, x1, len(new_clusters))
                if merged:
                    new_clusters.append(merged)
                    i += 2
                    changed = True
                    print(f"    Post-merge: x={x0} w_left={w_c} w_right={w_n} "
                          f"-> merged w={x1-x0}  "
                          f"({'both narrow' if both_narrow else 'sliver'})")
                    continue

            new_clusters.append(c)
            i += 1

        # Re-label
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


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_module1(image_path: str,
                out_dir: str             = "output_module1",
                white_noise_thresh: int  = 200,
                black_noise_thresh: int  = 200,
                mzs_threshold: float     = 3.0,
                gap_floor_ratio: float   = 0.45,
                save_individual_chars: bool = True,
                show_plots: bool         = False) -> dict:
    """
    Full Module 1 pipeline.

    Key parameters
    --------------
    gap_floor_ratio : Controls how deep a valley must be to count as a real
                      inter-character gap.  Range 0.0-1.0, default 0.45.
                      0.45 = valley must drop to <= 45% of peak projection.
                      RAISE for cleaner images (clearer gaps).  0.55-0.65.
                      LOWER for dense/damaged inscriptions.  0.30-0.40.
    mzs_threshold   : Modified Z-Score threshold for splitting wide segments.
                      Lower = more splits.  Default 3.0.
    """
    # Apply gap_floor_ratio to the module-level constant used by estimators
    import darken2 as _self
    _self.GAP_FLOOR_RATIO = gap_floor_ratio
    os.makedirs(out_dir, exist_ok=True)
    chars_dir = os.path.join(out_dir, "characters")
    os.makedirs(chars_dir, exist_ok=True)
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  Module 1  Brahmi Inscription Segmentation")
    print(f"  Input  : {image_path}")
    print(f"  Output : {out_dir}")
    print(sep)

    # Load
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    print(f"\n[0] Loaded  shape={raw.shape}")
    cv2.imwrite(os.path.join(out_dir, "00_raw.png"), raw)

    # Step 1
    print("\n[1] Preprocessing ...")
    gray, binary_clean, dilated_crop = preprocess(raw)
    cv2.imwrite(os.path.join(out_dir, "01a_gray.png"),         gray)
    cv2.imwrite(os.path.join(out_dir, "01b_binary_clean.png"), binary_clean)
    cv2.imwrite(os.path.join(out_dir, "01c_dilated_crop.png"), dilated_crop)

    # Step 2
    print("\n[2] Crop to inscription region ...")
    cropped, bbox = crop_to_inscription(dilated_crop, binary_clean)
    cv2.imwrite(os.path.join(out_dir, "02_cropped.png"), cropped)
    print(f"  Crop bbox: {bbox}")

    # Step 3
    print(f"\n[3] Noise removal (white={white_noise_thresh}, "
          f"black={black_noise_thresh}) ...")
    p1, inv, denoised = noise_removal(cropped, white_noise_thresh,
                                      black_noise_thresh)
    cv2.imwrite(os.path.join(out_dir, "03a_pass1.png"),    p1)
    cv2.imwrite(os.path.join(out_dir, "03b_inverted.png"), inv)
    cv2.imwrite(os.path.join(out_dir, "03c_denoised.png"), denoised)

    # Step 4
    print("\n[4] Baseline detection and flow analysis ...")
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
    print(f"  Boundaries: {boundaries}")

    # Step 8
    print("\n[8] Segment validation and MZS split ...")
    clusters = validate_and_split(
        boundaries, detail["proj_s"],
        rectified.shape[1], rectified,
        mzs_thresh=mzs_threshold
    )
    print(f"  After MZS split: {len(clusters)} segments")

    # Step 8b: Post-merge narrow over-splits
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

    if save_individual_chars:
        for num, crop, _ in chars:
            cv2.imwrite(os.path.join(chars_dir, f"char_{num:03d}.png"), crop)
        print(f"  Saved {len(chars)} chars -> {chars_dir}/")

    vis_pipeline([
        ("Gray",         gray),
        ("Binary",       binary_clean),
        ("Cropped",      cropped),
        ("Denoised",     denoised),
        ("Rectified",    rectified),
    ], os.path.join(out_dir, "pipeline_summary.png"))

    print(f"\n{'─'*64}")
    print(f"  CHARACTER COUNT  : {len(clusters)}")
    print(f"  Estimator votes  : proj={detail['proj']}  "
          f"cca={detail['cca']}  gaps={detail['gaps']}")
    print(f"  Confidence       : {conf.upper()}")
    print(f"  Flow type        : {baseline['flow_type'].upper()}")
    print(f"  Curvature        : {baseline['curvature']:.1f} px")
    print(f"  Tilt angle       : {baseline['angle_deg']:.1f} deg")
    print(f"  Gap floor used   : {gap_floor_ratio*100:.0f}% of max projection")
    print(f"  Output dir       : {out_dir}")
    print(f"{'─'*64}")
    print(f"  TIP: if count is wrong, adjust --gap_floor")
    print(f"    Too many chars? RAISE --gap_floor (e.g. 0.55)")
    print(f"    Too few chars?  LOWER --gap_floor (e.g. 0.35)")
    print(f"{'─'*64}\n")

    return {
        "chars":         chars,
        "count":         len(clusters),
        "confidence":    conf,
        "flow_type":     baseline["flow_type"],
        "baseline_info": baseline,
        "clusters":      clusters,
        "proj_detail":   detail,
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
            summary.append((img_path.name, r["count"],
                            r["confidence"], r["flow_type"]))
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}")
            summary.append((img_path.name, -1, "error", "?"))

    print("\n" + "=" * 62)
    print(f"{'Image':<32} {'Count':>6}  {'Conf':<8} {'Flow'}")
    print("-" * 62)
    for name, cnt, conf, flow in summary:
        print(f"{name:<32} {str(cnt) if cnt>=0 else 'FAIL':>6}  {conf:<8} {flow}")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Module 1 - Brahmi Inscription Segmentation")
    ap.add_argument("input",
                    help="Image file, or directory with --batch")
    ap.add_argument("--out",          default="output_module1")
    ap.add_argument("--white_thresh", type=int,   default=200)
    ap.add_argument("--black_thresh", type=int,   default=200)
    ap.add_argument("--mzs",          type=float, default=3.0,
                    help="Modified Z-Score threshold for wide-segment splitting (default 3.0)")
    ap.add_argument("--gap_floor",    type=float, default=0.45,
                    help="Gap-floor ratio 0-1 (default 0.45). "
                         "RAISE if too many chars, LOWER if too few.")
    ap.add_argument("--show",         action="store_true")
    ap.add_argument("--batch",        action="store_true")
    args = ap.parse_args()

    kw = dict(white_noise_thresh=args.white_thresh,
              black_noise_thresh=args.black_thresh,
              mzs_threshold=args.mzs,
              gap_floor_ratio=args.gap_floor,
              show_plots=args.show)

    if args.batch:
        batch_process(args.input, out_root=args.out, **kw)
    else:
        run_module1(args.input, out_dir=args.out, **kw)