import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as _ssim
import os
from pathlib import Path



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