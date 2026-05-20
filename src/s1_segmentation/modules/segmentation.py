import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as _ssim
import os
from pathlib import Path

from .counting import _estimate_min_char_width
from .counting import count_characters
from .baseline import detect_baseline, rectify
from .visualization import vis_count_signals



def force_split_massive_segments(clusters, binary_img, proj_s):
    """
    Final safety net: if a segment is massively wider than the median width,
    force-split it STRICTLY by the average character width.
    This ignores projection valleys entirely for these massive noise blocks
    and enforces a pure geometric cut.
    """
    if len(clusters) < 3:
        return clusters

    # 1. Calculate a highly robust median width
    # We strip the top and bottom 10% so tiny specks or giant noise blocks
    # don't ruin our average width calculation.
    widths = sorted([c["w"] for c in clusters])
    trim = max(1, len(widths) // 10)
    core_widths = widths[trim:-trim] if len(widths) > 4 else widths
    median_w = float(np.median(core_widths))

    new_clusters = []
    changed = False

    for c in clusters:
        x0 = c["x"]
        x1 = c["x"] + c["w"]
        w = c["w"]

        # If a box is >= 1.7x the median width, it's definitely multiple characters glued together.
        if w >= median_w * 1.7:
            # Calculate exactly how many characters should fit in this space
            expected_chars = max(2, int(round(w / median_w)))
            print(
                f"    Strict Split: Box at x={x0} is {w}px wide (median {median_w:.0f}px). Slicing evenly into {expected_chars}.")

            # Pure mathematical cuts based strictly on the average width
            chunk_width = w / expected_chars
            cuts = [int(x0 + i * chunk_width) for i in range(1, expected_chars)]

            boundaries = [x0] + cuts + [x1]
            for i in range(len(boundaries) - 1):
                nx0 = boundaries[i]
                nx1 = boundaries[i + 1]

                # Re-calculate the vertical bounding box (y and h) for this new chunk
                strip = binary_img[:, nx0:nx1]
                rows = np.where(np.any(strip == 255, axis=1))[0]
                if len(rows) > 0:
                    new_clusters.append({
                        "label": 0,  # Will re-label at end
                        "x": nx0, "y": int(rows[0]),
                        "w": nx1 - nx0, "h": int(rows[-1]) - int(rows[0]),
                        "area": int(np.sum(strip == 255))
                    })
            changed = True
        else:
            new_clusters.append(c)

    # Re-apply sequential labels (1, 2, 3...) if we sliced anything
    if changed:
        for i, c in enumerate(new_clusters):
            c["label"] = i + 1
        return new_clusters

    return clusters

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

    # ADD THIS LINE HERE:
    clusters = force_split_massive_segments(clusters, rectified, detail["proj_s"])

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