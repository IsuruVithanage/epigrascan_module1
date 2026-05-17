"""
Module 1: Image Preprocessing and Character Segmentation
for Ancient Brahmi Inscription Estampages

Pipeline:
  1. Grayscale conversion
  2. Non-Local Means denoising
  3. Adjusted Otsu thresholding (T_adjusted = T_global - 20)
  4. Binary inversion
  5. Morphological dilation (1×11 + 11×1)
  6. Largest-black-cluster crop
  7. Two-pass noise removal via CCA
  8. Cluster analysis and merging (3 stages)
  9. Character width analysis + Modified Z-Score outlier detection
 10. Character splitting and cropping
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────

def show(title, img, cmap="gray"):
    """Quick single-image display helper."""
    plt.figure(figsize=(12, 4))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_fig(path, img, title="", cmap="gray"):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────
# STEP 1 – INITIAL PREPROCESSING
# ─────────────────────────────────────────────────────────────

def step1_preprocess(raw_bgr):
    """
    Convert to grayscale → Non-Local Means denoising →
    Adjusted Otsu thresholding → binary inversion →
    two separate morphological operations:

      • dilated_for_crop  – aggressive 1×11 + 11×1 dilation used ONLY to
                            locate the inscription bounding box (step 2).
                            Never used for character work.
      • binary_refined    – gentle 3×3 morphological closing used for all
                            downstream character processing. Closes tiny
                            stroke-gap breaks without merging characters.

    ROOT CAUSE NOTE: applying the large dilation to the character processing
    chain washes out all detail (as seen in 01d_dilated.png). The large
    kernel is kept solely as a region-locator.

    Returns
    -------
    gray, denoised, binary_inv, dilated_for_crop, binary_refined
    """
    # 1a. Grayscale
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) if len(raw_bgr.shape) == 3 else raw_bgr.copy()

    # 1b. Non-Local Means denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10,
                                        templateWindowSize=7,
                                        searchWindowSize=21)

    # 1c. Adjusted Otsu thresholding: T_adjusted = T_global − 20
    t_global, _ = cv2.threshold(denoised, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_adjusted = max(0, t_global - 20)
    _, binary = cv2.threshold(denoised, t_adjusted, 255, cv2.THRESH_BINARY)
    print(f"  Otsu T_global={t_global:.1f}  T_adjusted={t_adjusted:.1f}")

    # 1d. Invert so foreground (text) is WHITE on BLACK.
    #     Auto-polarity: after Otsu, the majority of pixels should be
    #     background (black). If white pixels already dominate (> 55 %),
    #     the image is already in the correct polarity — skip inversion.
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio > 0.55:
        # Background is white → characters are black → invert
        binary_inv = cv2.bitwise_not(binary)
        print(f"  Polarity: inverted  (white_ratio before={white_ratio:.2f})")
    else:
        # Background already black → characters already white
        binary_inv = binary.copy()
        print(f"  Polarity: kept as-is (white_ratio before={white_ratio:.2f})")

    # 1e-CROP. Large dilation → ONLY for finding the inscription bounding box.
    #   1×11 horizontal + 11×1 vertical merges fragmented stroke clusters into
    #   one large blob so step2 can reliably detect the text region.
    se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    dilated_for_crop = cv2.dilate(binary_inv, se_h, iterations=1)
    dilated_for_crop = cv2.dilate(dilated_for_crop, se_v, iterations=1)

    # 1f-CHARACTER. Gentle 3×3 morphological closing → used for all character
    #   work. Closes tiny breaks in ink strokes without touching neighbour
    #   characters. (closing = dilate then erode, so net size change ≈ 0)
    se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_refined = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE,
                                      se_small, iterations=1)

    return gray, denoised, binary_inv, dilated_for_crop, binary_refined


# ─────────────────────────────────────────────────────────────
# STEP 2 – DETECT LARGEST BLACK CLUSTER → IRREGULAR CROP
# ─────────────────────────────────────────────────────────────

def step2_crop_largest_cluster(dilated_for_crop, binary_refined):
    """
    Use the aggressively-dilated image to locate the inscription bounding box,
    then crop the CLEAN binary_refined image to that region.

    This is the key fix: the large dilation merges scattered strokes into one
    detectable blob, giving us a reliable crop region — but we never use the
    washed-out dilated image for character work.

    Returns
    -------
    cropped_refined : binary_refined cropped to the inscription region
    bbox            : (x0, y0, x1, y1) in original image coordinates
    """
    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        dilated_for_crop, connectivity=8
    )
    if num_labels < 2:
        h, w = binary_refined.shape[:2]
        return binary_refined, (0, 0, w, h)

    # Largest foreground component from the dilated image → inscription region
    largest_label = 1 + np.argmax(stats_cc[1:, cv2.CC_STAT_AREA])
    x = stats_cc[largest_label, cv2.CC_STAT_LEFT]
    y = stats_cc[largest_label, cv2.CC_STAT_TOP]
    w = stats_cc[largest_label, cv2.CC_STAT_WIDTH]
    h = stats_cc[largest_label, cv2.CC_STAT_HEIGHT]

    pad = 8
    x0 = max(0, x - pad);  y0 = max(0, y - pad)
    x1 = min(binary_refined.shape[1], x + w + pad)
    y1 = min(binary_refined.shape[0], y + h + pad)

    # Crop the CLEAN (not dilated) image
    cropped_refined = binary_refined[y0:y1, x0:x1]
    return cropped_refined, (x0, y0, x1, y1)


# ─────────────────────────────────────────────────────────────
# STEP 3 – TWO-PASS NOISE REMOVAL
# ─────────────────────────────────────────────────────────────

def _remove_small_clusters(binary_img, threshold, target_color, replace_color):
    """
    CCA: remove connected components of `target_color` pixels
    whose area < threshold by replacing them with `replace_color`.
    """
    if target_color == 255:
        mask = binary_img.copy()
    else:
        mask = cv2.bitwise_not(binary_img)

    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = binary_img.copy()
    for lbl in range(1, num_labels):
        area = stats_cc[lbl, cv2.CC_STAT_AREA]
        if area < threshold:
            out[labels == lbl] = replace_color
    return out


def step3_two_pass_noise_removal(cropped_binary, white_threshold=300, black_threshold=300):
    """
    Pass 1: small WHITE clusters → BLACK (close minor gaps inside characters).
    Invert.
    Pass 2: small BLACK clusters → WHITE (remove residual speckle noise).
    Re-invert to restore polarity (WHITE text on BLACK).

    Thresholds are adjustable per image characteristics.
    """
    # Pass 1: white cluster removal
    pass1 = _remove_small_clusters(cropped_binary, white_threshold,
                                   target_color=255, replace_color=0)
    # Invert for pass 2
    inverted = cv2.bitwise_not(pass1)

    # Pass 2: black cluster removal (now black = noise after inversion)
    pass2 = _remove_small_clusters(inverted, black_threshold,
                                   target_color=255, replace_color=0)
    # Re-invert → WHITE text on BLACK
    final = cv2.bitwise_not(pass2)
    return pass1, inverted, final


# ─────────────────────────────────────────────────────────────
# STEP 4 – CLUSTER ANALYSIS AND MERGING
# ─────────────────────────────────────────────────────────────

def _get_clusters(binary_img):
    """
    Run CCA (8-connectivity) on WHITE pixels.
    Returns list of dicts with keys: label, x, y, w, h, area.
    """
    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    clusters = []
    for lbl in range(1, num_labels):
        x  = stats_cc[lbl, cv2.CC_STAT_LEFT]
        y  = stats_cc[lbl, cv2.CC_STAT_TOP]
        w  = stats_cc[lbl, cv2.CC_STAT_WIDTH]
        h  = stats_cc[lbl, cv2.CC_STAT_HEIGHT]
        area = stats_cc[lbl, cv2.CC_STAT_AREA]
        if area > 5:   # ignore micro-noise
            clusters.append({"label": lbl, "x": x, "y": y, "w": w, "h": h, "area": area})
    # Sort left-to-right
    clusters.sort(key=lambda c: c["x"])
    return clusters


def _x_overlap_ratio(a, b):
    """Fraction of the shorter span that overlaps with the other."""
    ax1, ax2 = a["x"], a["x"] + a["w"]
    bx1, bx2 = b["x"], b["x"] + b["w"]
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    shorter = min(ax2 - ax1, bx2 - bx1)
    return overlap / shorter if shorter > 0 else 0.0


def _y_overlap_ratio(a, b):
    ay1, ay2 = a["y"], a["y"] + a["h"]
    by1, by2 = b["y"], b["y"] + b["h"]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    shorter = min(ay2 - ay1, by2 - by1)
    return overlap / shorter if shorter > 0 else 0.0


def _merge_two(a, b):
    x0 = min(a["x"], b["x"])
    y0 = min(a["y"], b["y"])
    x1 = max(a["x"] + a["w"], b["x"] + b["w"])
    y1 = max(a["y"] + a["h"], b["y"] + b["h"])
    return {"label": a["label"], "x": x0, "y": y0,
            "w": x1 - x0, "h": y1 - y0,
            "area": a["area"] + b["area"]}


def _merge_pass(clusters, merge_fn):
    """Repeatedly apply merge_fn until no more merges occur."""
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if merge_fn(current, clusters[j]):
                    current = _merge_two(current, clusters[j])
                    used[j] = True
                    changed = True
            merged.append(current)
        clusters = merged
    clusters.sort(key=lambda c: c["x"])
    return clusters


def step4_cluster_analysis_and_merging(denoised_binary):
    """
    Three-stage merging strategy applied to CCA clusters.

    Pre-filter : discard micro-noise clusters (area < adaptive min_area).

    Stage 1 – Vertical Alignment : merge only SMALL fragments (< 1.5× median
              width) whose x-ranges overlap ≥ 0.7 with a neighbour.

    Stage 2 – Horizontal proximity : merge if x-gap ≤ 5% avg_width AND y-overlap ≥ 0.5.

    Stage 3 – Containment : merge if one bbox is ≥ 90% inside another.

    Fallback : if characters are physically connected (one giant cluster
               spans > 60% of image width), delegates to
               vertical_projection_segmentation().
    """
    clusters = _get_clusters(denoised_binary)
    if not clusters:
        return clusters

    # ── Pre-filter: drop pure noise ───────────────────────────
    areas = np.array([c["area"] for c in clusters])
    min_area = max(30, int(np.percentile(areas, 25) * 0.20))
    clusters = [c for c in clusters if c["area"] >= min_area]
    if not clusters:
        return clusters
    print(f"    Pre-filter min_area={min_area}  remaining={len(clusters)}")

    med_w = float(np.median([c["w"] for c in clusters]))

    # ── Stage 1: Vertical Alignment Merging ──────────────────
    def vertical_merge(a, b):
        is_frag_a = a["w"] < 1.5 * med_w
        is_frag_b = b["w"] < 1.5 * med_w
        if not (is_frag_a or is_frag_b):
            return False
        return _x_overlap_ratio(a, b) >= 0.7

    clusters = _merge_pass(clusters, vertical_merge)

    # ── Stage 2: Horizontal Merging ──────────────────────────
    avg_w = float(np.median([c["w"] for c in clusters]))
    gap_threshold = 0.05 * avg_w

    def horizontal_merge(a, b):
        h_gap = b["x"] - (a["x"] + a["w"])
        return h_gap <= gap_threshold and _y_overlap_ratio(a, b) >= 0.5

    clusters = _merge_pass(clusters, horizontal_merge)

    # ── Stage 3: Containment-Based Merging ───────────────────
    def containment_merge(a, b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"]+a["w"], a["y"]+a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area_b = b["w"] * b["h"]
        area_a = a["w"] * a["h"]
        if area_b > 0 and inter / area_b >= 0.9:
            return True
        if area_a > 0 and inter / area_a >= 0.9:
            return True
        return False

    clusters = _merge_pass(clusters, containment_merge)

    # ── Fallback: connected inscription band detected ─────────
    img_w = denoised_binary.shape[1]
    max_cluster_w = max((c["w"] for c in clusters), default=0)
    if len(clusters) <= 2 and max_cluster_w > 0.6 * img_w:
        print("    ⚠ Characters physically joined — using projection-profile segmentation")
        return vertical_projection_segmentation(denoised_binary)

    return clusters


# ─────────────────────────────────────────────────────────────
# VERTICAL PROJECTION PROFILE SEGMENTATION (fallback)
# ─────────────────────────────────────────────────────────────

def vertical_projection_segmentation(binary_img, smooth_sigma=3,
                                     valley_depth=0.20,
                                     min_char_w_ratio=0.025,
                                     max_char_w_ratio=0.22):
    """
    Segment characters from a single-line inscription using the column-wise
    white-pixel projection profile.  Robust against physically joined text.

    Returns list of cluster dicts (same schema as _get_clusters).
    Also returns the smoothed projection array for visualisation.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    h, w = binary_img.shape[:2]
    proj = np.sum(binary_img == 255, axis=0).astype(float)
    proj_s = gaussian_filter1d(proj, sigma=smooth_sigma)

    # Text column span
    threshold_col = proj_s.max() * 0.04
    nonzero = np.where(proj_s > threshold_col)[0]
    if len(nonzero) == 0:
        return [], proj_s
    t_start, t_end = int(nonzero[0]), int(nonzero[-1])

    min_w_px  = max(4, int(w * min_char_w_ratio))
    max_w_px  = int(w * max_char_w_ratio)

    # Find valleys (inverted peaks)
    inv = proj_s.max() - proj_s
    min_prom = proj_s.max() * valley_depth
    raw_peaks, _ = find_peaks(inv[t_start:t_end],
                              prominence=min_prom,
                              distance=min_w_px)
    valley_cols = (raw_peaks + t_start).tolist()

    # Build boundaries
    boundaries = sorted(set([t_start] + valley_cols + [t_end]))
    # Remove boundaries too close together
    clean = [boundaries[0]]
    for b in boundaries[1:]:
        if b - clean[-1] >= min_w_px:
            clean.append(b)
    boundaries = clean

    clusters = []
    for i in range(len(boundaries) - 1):
        x0, x1 = int(boundaries[i]), int(boundaries[i+1])
        seg_w = x1 - x0
        if seg_w < min_w_px:
            continue

        strip = binary_img[:, x0:x1]
        rows  = np.where(np.any(strip == 255, axis=1))[0]
        if len(rows) == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1])
        area = int(np.sum(strip == 255))

        # Sub-split over-wide segments at their lowest projection point
        if seg_w > max_w_px:
            sub = proj_s[x0:x1]
            mid = x0 + int(np.argmin(sub))
            for sx0, sx1 in [(x0, mid), (mid, x1)]:
                if sx1 - sx0 < min_w_px:
                    continue
                ss = binary_img[:, sx0:sx1]
                sr = np.where(np.any(ss == 255, axis=1))[0]
                if len(sr) == 0:
                    continue
                clusters.append({"label": len(clusters),
                                 "x": sx0, "y": int(sr[0]),
                                 "w": sx1-sx0, "h": int(sr[-1])-int(sr[0]),
                                 "area": int(np.sum(ss == 255))})
        else:
            clusters.append({"label": len(clusters),
                             "x": x0, "y": y0,
                             "w": seg_w, "h": y1-y0,
                             "area": area})

    clusters.sort(key=lambda c: c["x"])
    print(f"    Projection profile → {len(clusters)} segments  "
          f"(valleys={len(valley_cols)})")
    return clusters, proj_s




# ─────────────────────────────────────────────────────────────
# STEP 5 – CHARACTER WIDTH ANALYSIS + MODIFIED Z-SCORE
# ─────────────────────────────────────────────────────────────

def modified_z_score(widths):
    """
    MZS = 0.675 × |wi − M| / MAD
    where M  = median(widths)
          MAD = median(|wi − M|)
    """
    widths = np.array(widths, dtype=float)
    M   = np.median(widths)
    ADi = np.abs(widths - M)
    MAD = np.median(ADi)
    if MAD == 0:
        MAD = 1e-6
    mzs = 0.675 * ADi / MAD
    return mzs, M, MAD


def step5_split_outlier_clusters(clusters, threshold=3.0):
    """
    Detect merged-character clusters via Modified Z-Score on widths.
    Clusters with MZS > threshold are split at their horizontal midpoint.
    Returns expanded list of character bounding boxes.
    """
    widths = [c["w"] for c in clusters]
    if len(widths) < 3:
        return clusters   # not enough data for statistics

    mzs, M, MAD = modified_z_score(widths)
    print(f"  Width median={M:.1f}px  MAD={MAD:.1f}px")

    final_clusters = []
    for c, z in zip(clusters, mzs):
        if z > threshold:
            # Split at horizontal midpoint
            mid_x = c["x"] + c["w"] // 2
            left  = {"label": c["label"], "x": c["x"],   "y": c["y"],
                     "w": mid_x - c["x"],                "h": c["h"], "area": c["area"] // 2}
            right = {"label": c["label"], "x": mid_x,    "y": c["y"],
                     "w": c["x"] + c["w"] - mid_x,       "h": c["h"], "area": c["area"] // 2}
            final_clusters.extend([left, right])
            print(f"    Split cluster at x={c['x']} w={c['w']} MZS={z:.2f}")
        else:
            final_clusters.append(c)

    final_clusters.sort(key=lambda c: c["x"])
    return final_clusters, mzs, M, MAD


# ─────────────────────────────────────────────────────────────
# STEP 6 – CROP INDIVIDUAL CHARACTERS
# ─────────────────────────────────────────────────────────────

def step6_crop_characters(binary_img, clusters, padding=4):
    """
    Crop each character from the denoised binary image using bounding boxes.
    Returns list of (index, cropped_image, cluster_info).
    """
    chars = []
    h_img, w_img = binary_img.shape[:2]
    for i, c in enumerate(clusters):
        x0 = max(0, c["x"] - padding)
        y0 = max(0, c["y"] - padding)
        x1 = min(w_img, c["x"] + c["w"] + padding)
        y1 = min(h_img, c["y"] + c["h"] + padding)
        crop = binary_img[y0:y1, x0:x1]
        chars.append((i + 1, crop, c))
    return chars


# ─────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────

def visualise_pipeline(stages, out_dir):
    """Save a multi-panel summary of all preprocessing stages."""
    titles, imgs = zip(*stages)
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "pipeline_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Pipeline summary → {path}")


def visualise_clusters(binary_img, clusters, mzs, title, out_dir, filename):
    """Draw bounding boxes on binary image; red = outlier, green = normal."""
    vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for i, (c, z) in enumerate(zip(clusters, mzs)):
        color = (0, 0, 255) if z > 3.0 else (0, 200, 0)
        cv2.rectangle(vis, (c["x"], c["y"]),
                      (c["x"] + c["w"], c["y"] + c["h"]), color, 2)
        cv2.putText(vis, f"C{i+1}", (c["x"], max(c["y"] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, vis)
    print(f"  Cluster visualisation → {path}")
    return vis


def visualise_width_distribution(clusters, mzs, M, MAD, out_dir):
    """Replicate Figure 9 – width distribution with outlier markers."""
    widths = np.array([c["w"] for c in clusters])
    fig, ax = plt.subplots(figsize=(10, 4))

    # KDE curve
    if len(widths) > 2:
        xs = np.linspace(widths.min() - 20, widths.max() + 20, 300)
        kde = stats.gaussian_kde(widths)
        ax.plot(xs, kde(xs), "b-", lw=2, label="Normal Distribution (KDE)")

    # Vertical lines: normal = green, outlier = red
    for i, (c, z) in enumerate(zip(clusters, mzs)):
        color = "red" if z > 3.0 else "green"
        ls    = "--" if z > 3.0 else "-"
        ax.axvline(c["w"], color=color, linewidth=1.2, linestyle=ls,
                   label=f"Outlier ({c['w']}px)" if z > 3.0 else None)

    ax.axvline(M, color="black", lw=2, label=f"Median ({M:.0f}px)")
    ax.set_xlabel("Character Width (pixels)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Character Widths with Outliers")
    ax.annotate(f"MAD = {MAD:.1f} px", xy=(0.65, 0.85), xycoords="axes fraction", fontsize=11)
    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "width_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Width distribution → {path}")


def visualise_final_chars(chars, out_dir, max_display=20):
    """Grid view of all cropped characters."""
    n = min(len(chars), max_display)
    cols = min(n, 10)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")
    for idx, (num, crop, _) in enumerate(chars[:n]):
        axes[idx].imshow(crop, cmap="gray")
        axes[idx].set_title(f"Char {num}", fontsize=8)
    fig.suptitle("Segmented Characters", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "segmented_characters.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Character grid → {path}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_module1(image_path: str,
                out_dir: str = "output_module1",
                white_noise_thresh: int = 300,
                black_noise_thresh: int = 300,
                mzs_threshold: float = 3.0,
                save_individual_chars: bool = True,
                show_plots: bool = False) -> list:
    """
    Full Module 1 pipeline.

    Parameters
    ----------
    image_path          : path to raw estampage image
    out_dir             : directory for all output images
    white_noise_thresh  : pixel-area threshold for Pass-1 white cluster removal
    black_noise_thresh  : pixel-area threshold for Pass-2 black cluster removal
    mzs_threshold       : Modified Z-Score cutoff for merged-character detection (default 3.0)
    save_individual_chars: whether to save each character as a separate PNG
    show_plots          : whether to display matplotlib figures interactively

    Returns
    -------
    List of (index, cropped_char_image, cluster_info) tuples
    """
    os.makedirs(out_dir, exist_ok=True)
    chars_dir = os.path.join(out_dir, "characters")
    os.makedirs(chars_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Module 1 – Brahmi Inscription Segmentation")
    print(f"  Input : {image_path}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # ── Load ─────────────────────────────────────────────────
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    print(f"\n[0] Image loaded  shape={raw.shape}")
    cv2.imwrite(os.path.join(out_dir, "00_raw_input.png"), raw)

    # ── Step 1 – Preprocessing ───────────────────────────────
    print("\n[1] Initial Preprocessing …")
    gray, denoised, binary_inv, dilated_for_crop, binary_refined = step1_preprocess(raw)
    cv2.imwrite(os.path.join(out_dir, "01a_grayscale.png"),      gray)
    cv2.imwrite(os.path.join(out_dir, "01b_denoised.png"),       denoised)
    cv2.imwrite(os.path.join(out_dir, "01c_binary_inv.png"),     binary_inv)
    cv2.imwrite(os.path.join(out_dir, "01d_dilated_crop_only.png"), dilated_for_crop)
    cv2.imwrite(os.path.join(out_dir, "01e_binary_refined.png"), binary_refined)

    # ── Step 2 – Crop Largest Cluster ────────────────────────
    print("\n[2] Detecting largest cluster (from dilated) and cropping clean image …")
    cropped, bbox = step2_crop_largest_cluster(dilated_for_crop, binary_refined)
    cv2.imwrite(os.path.join(out_dir, "02_cropped_clean.png"), cropped)
    print(f"  Crop bbox: {bbox}")

    # ── Step 3 – Two-Pass Noise Removal ──────────────────────
    print(f"\n[3] Two-pass noise removal  "
          f"(white_thresh={white_noise_thresh}, black_thresh={black_noise_thresh}) …")
    pass1, inverted, pass2 = step3_two_pass_noise_removal(
        cropped, white_noise_thresh, black_noise_thresh
    )
    cv2.imwrite(os.path.join(out_dir, "03a_pass1_white_removal.png"), pass1)
    cv2.imwrite(os.path.join(out_dir, "03b_inverted.png"), inverted)
    cv2.imwrite(os.path.join(out_dir, "03c_pass2_black_removal.png"), pass2)

    # Use pass2 (denoised, WHITE text on BLACK) for all downstream steps
    denoised_binary = pass2

    # ── Step 4 – Cluster Analysis and Merging ────────────────
    print("\n[4] Cluster analysis and merging (3 stages) …")
    result4 = step4_cluster_analysis_and_merging(denoised_binary)

    # step4 may return (clusters, proj_profile) if projection fallback fired,
    # or just clusters when CCA merging succeeded.
    proj_profile = None
    if isinstance(result4, tuple):
        clusters, proj_profile = result4
    else:
        clusters = result4
    print(f"  Clusters after merging: {len(clusters)}")

    # ── Step 5 – Width Analysis + Outlier Detection ──────────
    print("\n[5] Width analysis + Modified Z-Score outlier detection …")
    if len(clusters) >= 3:
        result = step5_split_outlier_clusters(clusters, threshold=mzs_threshold)
        if isinstance(result, tuple):
            clusters_final, mzs_scores, M, MAD = result[0], result[1], result[2], result[3]
        else:
            clusters_final = result
            widths = [c["w"] for c in clusters_final]
            mzs_scores, M, MAD = modified_z_score(widths)
    else:
        clusters_final = clusters
        widths = [c["w"] for c in clusters_final]
        mzs_scores = np.zeros(len(widths))
        M = np.median(widths) if widths else 0
        MAD = 0

    print(f"  Characters after splitting: {len(clusters_final)}")

    # Recompute MZS for the final list for visualisation
    widths_final = [c["w"] for c in clusters_final]
    mzs_final, M_f, MAD_f = modified_z_score(widths_final) if len(widths_final) >= 3 \
        else (np.zeros(len(widths_final)), 0, 0)

    # ── Visualisations ───────────────────────────────────────
    print("\n[6] Generating visualisations …")

    vis_clusters = visualise_clusters(
        denoised_binary, clusters_final, mzs_final,
        "Segmented Characters", out_dir, "04_cluster_boxes.png"
    )

    visualise_width_distribution(clusters_final, mzs_final, M_f, MAD_f, out_dir)

    # Optional: projection profile plot (when fallback method was used)
    if proj_profile is not None:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(proj_profile, color="steelblue", lw=1.5)
        for c in clusters_final:
            ax.axvline(c["x"],           color="green", lw=1, ls="--", alpha=0.7)
            ax.axvline(c["x"] + c["w"],  color="red",   lw=1, ls="--", alpha=0.7)
        ax.set_xlabel("Column index (x)"); ax.set_ylabel("White pixel count")
        ax.set_title("Vertical Projection Profile  (green=seg start, red=seg end)")
        fig.tight_layout()
        pp_path = os.path.join(out_dir, "04b_projection_profile.png")
        fig.savefig(pp_path, dpi=150); plt.close(fig)
        print(f"  Projection profile → {pp_path}")

    pipeline_stages = [
        ("Input (grayscale)", gray),
        ("Denoised", denoised),
        ("Binary + inverted", binary_inv),
        ("Refined (3×3 closing)", binary_refined),
        ("Dilated [crop-only]", dilated_for_crop),
        ("Cropped (clean)", cropped),
        ("Pass-1 noise removal", pass1),
        ("Inverted", inverted),
        ("Pass-2 noise removal", pass2),
    ]
    visualise_pipeline(pipeline_stages, out_dir)

    # ── Step 6 – Crop Characters ─────────────────────────────
    print("\n[7] Cropping individual characters …")
    chars = step6_crop_characters(denoised_binary, clusters_final)
    visualise_final_chars(chars, out_dir)

    if save_individual_chars:
        for num, crop, _ in chars:
            p = os.path.join(chars_dir, f"char_{num:03d}.png")
            cv2.imwrite(p, crop)
        print(f"  Saved {len(chars)} individual characters → {chars_dir}/")

    if show_plots:
        for title, img in pipeline_stages:
            show(title, img)
        plt.figure(figsize=(14, 4))
        plt.imshow(cv2.cvtColor(vis_clusters, cv2.COLOR_BGR2RGB))
        plt.title("Segmented Character Boxes")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print(f"\n✓ Module 1 complete. {len(chars)} characters segmented.")
    print(f"  All outputs in: {out_dir}\n")
    return chars


# ─────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────

def batch_process(input_dir: str, out_root: str = "batch_output", **kwargs):
    """
    Process all images in a directory.
    Supported formats: .png, .jpg, .jpeg, .tif, .tiff, .bmp
    """
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = [p for p in Path(input_dir).iterdir()
              if p.suffix.lower() in exts]
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} image(s) in {input_dir}")
    all_results = {}
    for img_path in sorted(images):
        out_dir = os.path.join(out_root, img_path.stem)
        try:
            chars = run_module1(str(img_path), out_dir=out_dir, **kwargs)
            all_results[img_path.name] = len(chars)
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")
            all_results[img_path.name] = -1

    print("\n" + "=" * 40)
    print("Batch Summary")
    print("=" * 40)
    for name, n in all_results.items():
        status = f"{n} chars" if n >= 0 else "FAILED"
        print(f"  {name:30s} → {status}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage examples:
      # Single image
      python module1_brahmi_segmentation.py path/to/estampage.jpg

      # Single image with custom thresholds
      python module1_brahmi_segmentation.py path/to/estampage.jpg \
          --out output_dir \
          --white_thresh 200 \
          --black_thresh 150 \
          --mzs 3.0

      # Batch (directory of images)
      python module1_brahmi_segmentation.py path/to/images/ --batch
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Module 1: Brahmi Inscription Preprocessing & Segmentation"
    )
    parser.add_argument("input", help="Path to image file OR directory (with --batch)")
    parser.add_argument("--out",          default="output_module1",
                        help="Output directory (default: output_module1)")
    parser.add_argument("--white_thresh", type=int, default=300,
                        help="White cluster noise removal threshold (default: 300 px²)")
    parser.add_argument("--black_thresh", type=int, default=300,
                        help="Black cluster noise removal threshold (default: 300 px²)")
    parser.add_argument("--mzs",          type=float, default=3.0,
                        help="Modified Z-Score threshold for outlier detection (default: 3.0)")
    parser.add_argument("--show",         action="store_true",
                        help="Show matplotlib plots interactively")
    parser.add_argument("--batch",        action="store_true",
                        help="Process all images in the input directory")

    args = parser.parse_args()

    if args.batch:
        batch_process(
            args.input, out_root=args.out,
            white_noise_thresh=args.white_thresh,
            black_noise_thresh=args.black_thresh,
            mzs_threshold=args.mzs,
            show_plots=args.show
        )
    else:
        run_module1(
            args.input,
            out_dir=args.out,
            white_noise_thresh=args.white_thresh,
            black_noise_thresh=args.black_thresh,
            mzs_threshold=args.mzs,
            show_plots=args.show
        )