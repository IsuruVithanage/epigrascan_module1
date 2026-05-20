import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

# Import the refactored modular pipeline
from src.s1_segmentation.modules.preprocessing import preprocess, crop_to_inscription, extract_character_band, noise_removal, remove_border_blobs
from src.s1_segmentation.modules.baseline import detect_baseline, rectify
from src.s1_segmentation.modules.counting import count_characters
from src.s1_segmentation.modules.segmentation import place_boundaries, filter_weak_boundaries, validate_and_split, post_merge_narrow_segments, force_split_massive_segments, crop_characters, detect_text_rows, segment_one_row
from src.s1_segmentation.modules.visualization import vis_baseline, vis_count_signals, vis_segmentation, vis_chars_grid, vis_pipeline
from src.s1_segmentation.modules.calibration import auto_calibrate

# Since we modularized the code, we need to inject GAP_FLOOR_RATIO into the modules
# that use globals().get("GAP_FLOOR_RATIO") or we can just patch sys.modules.
# For exact compatibility without changing their logic, we can inject into their namespaces.
import src.s1_segmentation.modules.counting as _counting
import src.s1_segmentation.modules.segmentation as _segmentation

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

    # Inject GAP_FLOOR_RATIO to submodules because original code uses globals().get("GAP_FLOOR_RATIO")
    _counting.GAP_FLOOR_RATIO = _gap_floor
    _segmentation.GAP_FLOOR_RATIO = _gap_floor

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

        # Step 8c
        print("\n[8c] Force-splitting massive merged segments ...")
        clusters = force_split_massive_segments(clusters, rectified, detail["proj_s"])
        print(f"  Count after force-split: {len(clusters)}")

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Module 1 - Brahmi Inscription Preprocessing & Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input",
                    nargs="?",
                    help="Image file, or directory (use --batch)",
                    default="../../data/raw_estampages/"
                    )
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
    ap.add_argument("--batch",   action="store_true", help="Process whole directory.", default=True)
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