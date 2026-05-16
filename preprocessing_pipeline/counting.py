import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as _ssim
import os
from pathlib import Path

from .baseline import _get_char_centroids

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