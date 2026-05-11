"""
Module 1: Image Preprocessing and Character Segmentation
Focus: Exact Character Sequencing & Safe Resolution Scaling
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
import os
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# UTILITY & VISUALIZATION
# ─────────────────────────────────────────────────────────────

def visualise_pipeline(stages, out_dir):
    titles, imgs = zip(*stages)
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1: axes = [axes]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "00_pipeline_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

def visualise_final_chars(chars, out_dir, max_display=30):
    n = min(len(chars), max_display)
    if n == 0: return
    cols = min(n, 10)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for ax in axes: ax.axis("off")
    for idx, (num, crop, _) in enumerate(chars[:n]):
        axes[idx].imshow(crop, cmap="gray")
        axes[idx].set_title(f"Char {num}", fontsize=8)
    fig.suptitle("Segmented Characters (Correct Reading Order)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "05_segmented_characters.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# STEP 1 to 3 – SAFE PREPROCESSING & NOISE REMOVAL
# ─────────────────────────────────────────────────────────────

def step1_preprocess(raw_bgr):
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) if len(raw_bgr.shape) == 3 else raw_bgr.copy()
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    t_global, _ = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_adjusted = max(0, t_global - 20)
    _, binary = cv2.threshold(denoised, t_adjusted, 255, cv2.THRESH_BINARY)

    # 🟢 FIX: Restored safe polarity logic
    white_ratio = np.sum(binary == 255) / binary.size
    binary_inv = cv2.bitwise_not(binary) if white_ratio > 0.55 else binary.copy()

    se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_refined = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, se_small, iterations=1)

    return gray, denoised, binary_inv, binary_refined

def _remove_small_clusters(binary_img, threshold, target_color, replace_color):
    mask = binary_img.copy() if target_color == 255 else cv2.bitwise_not(binary_img)
    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = binary_img.copy()
    for lbl in range(1, num_labels):
        if stats_cc[lbl, cv2.CC_STAT_AREA] < threshold:
            out[labels == lbl] = replace_color
    return out

def step3_two_pass_noise_removal(cropped_binary, white_thresh=100, black_thresh=100):
    pass1 = _remove_small_clusters(cropped_binary, white_thresh, 255, 0)
    inverted = cv2.bitwise_not(pass1)
    pass2 = _remove_small_clusters(inverted, black_thresh, 255, 0)
    return cv2.bitwise_not(pass2)

# ─────────────────────────────────────────────────────────────
# STEP 4 – SCALE-INVARIANT LINE DETECTION
# ─────────────────────────────────────────────────────────────

def step4_scale_invariant_line_segmentation(denoised_binary):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(denoised_binary, connectivity=8)

    # 🟢 THE DUST BLINDERS: Ignore anything smaller than 150 pixels when calculating scale!
    # This guarantees it measures the real letters, not the background dust.
    valid_widths = [stats[i, cv2.CC_STAT_WIDTH] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 150]

    dynamic_w = int(np.median(valid_widths)) if valid_widths else 40
    smear_len = int(dynamic_w * 2.5)

    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (smear_len, 1))
    smeared = cv2.dilate(denoised_binary, line_kernel, iterations=2)

    line_contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in line_contours:
        lx, ly, lw, lh = cv2.boundingRect(cnt)
        if lh > 10 and lw > dynamic_w:
            lines.append((lx, ly, lw, lh))

    lines.sort(key=lambda b: b[1])
    return lines, smeared

# ─────────────────────────────────────────────────────────────
# STEP 5 & 6 – CCA CLUSTERING & K-MEANS SPLITTING
# ─────────────────────────────────────────────────────────────

def _get_clusters(binary_img):
    num_labels, _, stats_cc, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    clusters = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats_cc[lbl, cv2.CC_STAT_LEFT], stats_cc[lbl, cv2.CC_STAT_TOP], stats_cc[lbl, cv2.CC_STAT_WIDTH], stats_cc[lbl, cv2.CC_STAT_HEIGHT], stats_cc[lbl, cv2.CC_STAT_AREA]

        solidity = area / (w * h) if w * h > 0 else 0

        # 🟢 Filter noise out of the actual character list
        if area > 30 and solidity >= 0.12:
            clusters.append({"label": lbl, "x": x, "y": y, "w": w, "h": h, "area": area, "solidity": solidity})

    clusters.sort(key=lambda c: c["x"])
    return clusters

def _merge_pass(clusters, merge_fn):
    changed = True
    while changed:
        changed = False
        merged, used = [], [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]: continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]: continue
                if merge_fn(current, clusters[j]):
                    current = {"label": current["label"],
                               "x": min(current["x"], clusters[j]["x"]),
                               "y": min(current["y"], clusters[j]["y"]),
                               "w": max(current["x"]+current["w"], clusters[j]["x"]+clusters[j]["w"]) - min(current["x"], clusters[j]["x"]),
                               "h": max(current["y"]+current["h"], clusters[j]["y"]+clusters[j]["h"]) - min(current["y"], clusters[j]["y"]),
                               "area": current["area"] + clusters[j]["area"],
                               "solidity": current.get("solidity", 1.0)}
                    used[j], changed = True, True
            merged.append(current)
        clusters = merged
    clusters.sort(key=lambda c: c["x"])
    return clusters

def step5_local_cluster_merging(line_binary):
    clusters = _get_clusters(line_binary)
    if not clusters: return clusters

    areas = np.array([c["area"] for c in clusters])
    min_area = max(20, int(np.percentile(areas, 25) * 0.20))
    clusters = [c for c in clusters if c["area"] >= min_area]
    if not clusters: return clusters

    med_w = float(np.median([c["w"] for c in clusters]))

    clusters = _merge_pass(clusters, lambda a, b: (a["w"] < 1.5 * med_w or b["w"] < 1.5 * med_w) and (max(0, min(a["x"]+a["w"], b["x"]+b["w"]) - max(a["x"], b["x"])) / min(a["w"], b["w"]) >= 0.7 if min(a["w"], b["w"])>0 else False))
    avg_w = float(np.median([c["w"] for c in clusters]))
    clusters = _merge_pass(clusters, lambda a, b: (b["x"] - (a["x"] + a["w"]) <= 0.05 * avg_w) and (max(0, min(a["y"]+a["h"], b["y"]+b["h"]) - max(a["y"], b["y"])) / min(a["h"], b["h"]) >= 0.5 if min(a["h"], b["h"])>0 else False))

    def containment_merge(a, b):
        inter = max(0, min(a["x"]+a["w"], b["x"]+b["w"]) - max(a["x"], b["x"])) * max(0, min(a["y"]+a["h"], b["y"]+b["h"]) - max(a["y"], b["y"]))
        return inter / (b["w"]*b["h"]) >= 0.9 if (b["w"]*b["h"])>0 else False or inter / (a["w"]*a["h"]) >= 0.9 if (a["w"]*a["h"])>0 else False

    return _merge_pass(clusters, containment_merge)

def step6_kmeans_split_outliers(clusters):
    if not clusters: return []
    widths = np.array([c["w"] for c in clusters]).reshape(-1, 1)

    if len(widths) < 4:
        for c in clusters: c["split_warning"] = False
        return clusters

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(widths)
    centers = kmeans.cluster_centers_.flatten()

    normal_label = np.argmin(centers)
    fused_label = np.argmax(centers)
    normal_width = centers[normal_label]

    final_clusters = []
    for c, label in zip(clusters, kmeans.labels_):
        if label == fused_label and c["w"] > (normal_width * 1.5):
            num_letters = max(2, round(c["w"] / normal_width))
            slice_w = c["w"] // num_letters
            for i in range(num_letters):
                final_clusters.append({
                    "label": c["label"], "x": c["x"] + (i * slice_w), "y": c["y"],
                    "w": slice_w, "h": c["h"], "area": c["area"] // num_letters,
                    "solidity": c.get("solidity", 1.0), "split_warning": True
                })
        else:
            c["split_warning"] = False
            final_clusters.append(c)

    final_clusters.sort(key=lambda c: c["x"])
    return final_clusters

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE ARCHITECTURE
# ─────────────────────────────────────────────────────────────

def run_module1(image_path: str, out_dir: str = "output_module1", white_noise_thresh: int = 100, black_noise_thresh: int = 100):
    os.makedirs(out_dir, exist_ok=True)
    chars_dir = os.path.join(out_dir, "characters")
    os.makedirs(chars_dir, exist_ok=True)

    print(f"\n{'='*60}\n  Module 1 – Exact-Count Segmentation Engine\n  Input: {Path(image_path).name}\n{'='*60}")

    if not os.path.exists(image_path):
        print(f"  [!] Error: File missing. Skipping.")
        return []

    raw = cv2.imread(image_path)
    if raw is None:
        print(f"  [!] Error: OpenCV could not decode '{Path(image_path).name}'. Skipping.")
        return []

    gray, denoised, binary_inv, binary_refined = step1_preprocess(raw)

    binary_refined[0:5, :] = 0; binary_refined[-5:, :] = 0; binary_refined[:, 0:5] = 0; binary_refined[:, -5:] = 0

    denoised_binary = step3_two_pass_noise_removal(binary_refined, white_noise_thresh, black_noise_thresh)

    print("\n[Stage 1] Executing Scale-Invariant Line Segregation...")
    lines, smeared = step4_scale_invariant_line_segmentation(denoised_binary)
    cv2.imwrite(os.path.join(out_dir, "02_dynamic_smear_xray.png"), smeared)

    print("\n[Stage 2] Executing K-Means Cluster Splitting & Mapping...")
    global_clusters = []
    preview_img = cv2.cvtColor(denoised_binary, cv2.COLOR_GRAY2BGR)

    for line_idx, (lx, ly, lw, lh) in enumerate(lines):
        line_roi = denoised_binary[ly:ly+lh, lx:lx+lw]
        local_merged = step5_local_cluster_merging(line_roi)
        local_cleaved = step6_kmeans_split_outliers(local_merged)

        centroids = []
        for c in local_cleaved:
            gx, gy, gw, gh = lx + c["x"], ly + c["y"], c["w"], c["h"]
            c["global_x"], c["global_y"] = gx, gy
            global_clusters.append(c)

            color = (0, 165, 255) if c.get("split_warning") else (0, 255, 0)
            cv2.rectangle(preview_img, (gx, gy), (gx + gw, gy + gh), color, 2)

            cx, cy = gx + (gw // 2), gy + (gh // 2)
            centroids.append((cx, cy))
            cv2.circle(preview_img, (cx, cy), 4, (0, 0, 255), -1)

        if len(centroids) >= 2:
            centroids_np = np.array(centroids)
            x_pts, y_pts = centroids_np[:, 0], centroids_np[:, 1]
            _, unique_idx = np.unique(x_pts, return_index=True)
            unique_idx.sort()

            if len(unique_idx) >= 2:
                x_pts, y_pts = x_pts[unique_idx], y_pts[unique_idx]
                k_val = min(3, len(unique_idx) - 1)
                x_smooth = np.linspace(x_pts.min(), x_pts.max(), 300)
                y_smooth = make_interp_spline(x_pts, y_pts, k=k_val)(x_smooth)
                curve_pts = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
                cv2.polylines(preview_img, [curve_pts], isClosed=False, color=(255, 0, 0), thickness=2)

    cv2.imwrite(os.path.join(out_dir, "03_spline_mapped_bounds.png"), preview_img)

    print("\n[Stage 3] Formatting final characters & Exporting Metadata...")
    chars = []
    metadata = {}

    for i, c in enumerate(global_clusters):
        crop = denoised_binary[c["global_y"]:c["global_y"]+c["h"], c["global_x"]:c["global_x"]+c["w"]]
        if crop.size > 0:
            margin = int(max(c["h"], c["w"]) * 0.15)
            diff = abs(c["h"] - c["w"])
            pt, pb, pl, pr = margin, margin, margin, margin
            if c["h"] > c["w"]: pl += diff//2; pr += diff - diff//2
            else: pt += diff//2; pb += diff - diff//2

            sq_crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
            final_crop = cv2.resize(sq_crop, (224, 224))

            char_filename = f"char_{i+1:03d}.png"
            chars.append((i+1, final_crop, c))
            cv2.imwrite(os.path.join(chars_dir, char_filename), final_crop)

            warnings = []
            if c.get("split_warning"): warnings.append("Splitted by K-Means")
            if c.get("solidity", 1.0) < 0.20: warnings.append("Low Solidity (Potential Rock Scratch)")

            metadata[char_filename] = {
                "reading_order_idx": i + 1,
                "warnings": warnings,
                "box": {"x": c["global_x"], "y": c["global_y"], "w": c["w"], "h": c["h"]}
            }

    with open(os.path.join(out_dir, "module1_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    visualise_pipeline([("Raw", gray), ("Denoised", denoised), ("Binary", binary_inv), ("Smear Map", smeared)], out_dir)
    visualise_final_chars(chars, out_dir)

    print(f"✓ Extracted {len(chars)} characters.")
    print(f"✓ Metadata generated: {os.path.join(out_dir, 'module1_metadata.json')}")
    return chars

# ─────────────────────────────────────────────────────────────
# BATCH ENTRY POINT
# ─────────────────────────────────────────────────────────────

def batch_process(input_dir, out_root="batch_output", **kwargs):
    images = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and not p.name.startswith(".")]
    if not images:
        print(f"No valid images found in {input_dir}")
        return

    for img_path in sorted(images):
        try:
            # 🟢 FIX: Safe middle-ground thresholds passed to the runner
            run_module1(str(img_path), out_dir=os.path.join(out_root, img_path.stem), white_noise_thresh=100, black_noise_thresh=100)
        except Exception as e:
            print(f"  [!] Fatal error processing {img_path.name}: {e}\n  Skipping...")
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--out", default="output_module1")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--white_thresh", type=int, default=100)
    parser.add_argument("--black_thresh", type=int, default=100)
    args = parser.parse_args()

    if args.batch: batch_process(args.input, out_root=args.out, white_noise_thresh=args.white_thresh, black_noise_thresh=args.black_thresh)
    else: run_module1(args.input, out_dir=args.out, white_noise_thresh=args.white_thresh, black_noise_thresh=args.black_thresh)