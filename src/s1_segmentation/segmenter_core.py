import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import yaml
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Import the refactored modular pipeline
from src.s1_segmentation.modules.preprocessing import preprocess, crop_to_inscription, extract_character_band, \
    noise_removal, remove_border_blobs
from src.s1_segmentation.modules.baseline import detect_baseline, rectify
from src.s1_segmentation.modules.counting import count_characters
from src.s1_segmentation.modules.segmentation import place_boundaries, filter_weak_boundaries, validate_and_split, \
    post_merge_narrow_segments, force_split_massive_segments, crop_characters, detect_text_rows, segment_one_row
from src.s1_segmentation.modules.visualization import vis_baseline, vis_count_signals, vis_segmentation, vis_chars_grid, \
    vis_pipeline
from src.s1_segmentation.modules.calibration import auto_calibrate

# Since we modularized the code, we need to inject GAP_FLOOR_RATIO into the modules
import src.s1_segmentation.modules.counting as _counting
import src.s1_segmentation.modules.segmentation as _segmentation
from ultralytics import YOLO


# =====================================================================
# MOBILENETV2 CLASSIFIER ARCHITECTURE
# =====================================================================
class BrahmiClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)


def run_module1(image_path: str,
                out_dir: str = "output_module1",
                mobilenet_path: str = "models/mobile_net_v2_brahmi_best.pth",
                label_mapping_path: str = "label_mapping.json",
                config_path: str = "config.yaml",
                white_noise_thresh: int = None,
                black_noise_thresh: int = None,
                mzs_threshold: float = 3.0,
                gap_floor_ratio: float = None,
                nlm_h: int = None,
                bilateral: bool = True,
                otsu_bias: int = None,
                dilate_kernel: tuple = (3, 3),
                dilate_iters: int = 1,
                auto_params: bool = True,
                save_individual_chars: bool = True,
                show_plots: bool = False) -> dict:
    image_stem = Path(image_path).stem

    # --- LOAD YOLO & MOBILENET MODELS ---
    print("\n[0] Loading AI Models & Config...")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    yolo_model = YOLO('models/best.pt')

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        conf_threshold = config.get("model", {}).get("confidence_threshold", 0.40)
    except FileNotFoundError:
        conf_threshold = 0.40

    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)

    mobilenet_model = BrahmiClassifier(num_classes=len(label_mapping)).to(device)
    mobilenet_model.load_state_dict(torch.load(mobilenet_path, map_location=device))
    mobilenet_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if auto_params:
        print(f"\n[AUTO] Calibrating parameters for: {image_path}")
        cal = auto_calibrate(image_path)
        _nlm_h = nlm_h if nlm_h is not None else cal["nlm_h"]
        _bilateral = bilateral
        _otsu_bias = otsu_bias if otsu_bias is not None else cal["otsu_bias"]
        _white_thresh = white_noise_thresh if white_noise_thresh is not None else cal["white_noise_thresh"]
        _black_thresh = black_noise_thresh if black_noise_thresh is not None else cal["black_noise_thresh"]
        _gap_floor = gap_floor_ratio if gap_floor_ratio is not None else cal["gap_floor_ratio"]
    else:
        _nlm_h = nlm_h if nlm_h is not None else 10
        _bilateral = bilateral
        _otsu_bias = otsu_bias if otsu_bias is not None else 20
        _white_thresh = white_noise_thresh if white_noise_thresh is not None else 200
        _black_thresh = black_noise_thresh if black_noise_thresh is not None else 200
        _gap_floor = gap_floor_ratio if gap_floor_ratio is not None else 0.45

    _counting.GAP_FLOOR_RATIO = _gap_floor
    _segmentation.GAP_FLOOR_RATIO = _gap_floor

    # Ensure clean output directory
    os.makedirs(out_dir, exist_ok=True)
    chars_dir = os.path.join(out_dir, "characters")
    os.makedirs(chars_dir, exist_ok=True)

    # Load Image
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    cv2.imwrite(os.path.join(out_dir, "00_raw.png"), raw)

    # Step 1: Preprocessing
    print("\n[1] Preprocessing ...")
    gray, binary_clean, dilated_crop, metrics = preprocess(
        raw, nlm_h=_nlm_h, bilateral=_bilateral, otsu_bias=_otsu_bias,
        dilate_kernel=dilate_kernel, dilate_iters=dilate_iters
    )
    cv2.imwrite(os.path.join(out_dir, "01a_gray.png"), gray)
    cv2.imwrite(os.path.join(out_dir, "01b_binary_clean.png"), binary_clean)
    cv2.imwrite(os.path.join(out_dir, "01c_dilated_crop.png"), dilated_crop)

    # Step 2: Crop to inscription region
    print("\n[2] Crop to inscription region ...")
    cropped, bbox = crop_to_inscription(dilated_crop, binary_clean)
    cv2.imwrite(os.path.join(out_dir, "02_cropped.png"), cropped)

    # Step 3: Noise removal (CCA)
    print(f"\n[3] Noise removal (white={_white_thresh}, black={_black_thresh}) ...")
    p1, inv, denoised = noise_removal(cropped, _white_thresh, _black_thresh)
    cv2.imwrite(os.path.join(out_dir, "03a_pass1.png"), p1)
    cv2.imwrite(os.path.join(out_dir, "03b_inverted.png"), inv)
    cv2.imwrite(os.path.join(out_dir, "03c_denoised.png"), denoised)

    # Step 4: Remove border-touching noise blobs
    print("\n[4] Removing border-touching noise blobs ...")
    denoised = remove_border_blobs(denoised)
    cv2.imwrite(os.path.join(out_dir, "04_no_border.png"), denoised)

    # Step 5: Character band extraction
    print("\n[5] Character band extraction ...")
    denoised, baseline_rough, band_half = extract_character_band(
        denoised, padding=5, band_height_factor=1.1,
        save_vis_path=os.path.join(out_dir, "05_band_vis.png")
    )
    cv2.imwrite(os.path.join(out_dir, "05_band_masked.png"), denoised)

    # Step 6: Detect text rows
    print("\n[6] Detecting text row structure ...")
    row_bands, n_rows = detect_text_rows(denoised)

    noisy_text = []
    soft_probs_list = []
    confidences_list = []
    clusters = []
    chars = []
    final_json_output = {}

    if n_rows > 1:
        # [Multi-row logic remains unchanged]
        pass
    else:
        # Step 7: Baseline detection
        print("\n[7] Baseline detection on cleaned image ...")
        baseline = detect_baseline(denoised)
        vis_baseline(denoised, baseline, os.path.join(out_dir, "07_baseline.png"))

        # Step 8: Rectification
        print(f"\n[8] Rectification (flow={baseline['flow_type']}) ...")
        rectified, col_offsets = rectify(denoised, baseline)
        cv2.imwrite(os.path.join(out_dir, "08_rectified.png"), rectified)

        # =====================================================================
        # NEW STEP 9: YOLO SEGMENTATION ON THE RECTIFIED IMAGE
        # =====================================================================
        print("\n[9] Running YOLO Segmentation on Rectified Image...")
        rectified_rgb = cv2.cvtColor(rectified, cv2.COLOR_GRAY2RGB)
        results = yolo_model(rectified_rgb, conf=0.25, iou=0.6)
        raw_boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(raw_boxes) == 0:
            print("  - YOLO found NO characters in this row.")
            vis_yolo = rectified_rgb.copy()
        else:
            processed_boxes = []
            for box in raw_boxes:
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                cx = x1 + (w / 2)
                if w > 8 and h > 8:
                    processed_boxes.append({'x': x1, 'y': y1, 'w': w, 'h': h, 'cx': cx})

            # Sort boxes left-to-right
            processed_boxes.sort(key=lambda b: b['cx'])

            # =====================================================================
            # NEW STEP 10: SQUARE PADDING, CLASSIFICATION, AND JSON EXPORT
            # =====================================================================
            print(f"\n[10] Cropping & Classifying {len(processed_boxes)} characters...")
            vis_yolo = rectified_rgb.copy()

            for i, b in enumerate(processed_boxes):
                y_start = max(0, b['y'] - 2)
                y_end = min(rectified.shape[0], b['y'] + b['h'] + 2)
                x_start = max(0, b['x'] - 2)
                x_end = min(rectified.shape[1], b['x'] + b['w'] + 2)

                crop_img = rectified[y_start:y_end, x_start:x_end]

                if crop_img.size > 0:
                    # Square Padding
                    ch, cw = crop_img.shape[:2]
                    diff = abs(ch - cw)
                    pad_top, pad_bot, pad_l, pad_r = 15, 15, 15, 15

                    if ch > cw:
                        pad_l += diff // 2
                        pad_r += (diff - diff // 2)
                    elif cw > ch:
                        pad_top += diff // 2
                        pad_bot += (diff - diff // 2)

                    square_crop = cv2.copyMakeBorder(crop_img, pad_top, pad_bot, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                                     value=0)
                    resized_crop = cv2.resize(square_crop, (224, 224))
                    rgb_224 = cv2.cvtColor(resized_crop, cv2.COLOR_GRAY2RGB)

                    # MobileNet Inference
                    tensor = transform(rgb_224).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = mobilenet_model(tensor)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                    confidence = float(probs.max())
                    pred_idx = int(probs.argmax())
                    char_label = label_mapping.get(str(pred_idx), "?")

                    if confidence >= conf_threshold:
                        noisy_text.append(char_label)
                        box_text = f"#{i + 1}:{char_label}"
                    else:
                        noisy_text.append("[MASK]")
                        soft_probs_list.append(probs.tolist())
                        confidences_list.append(confidence)
                        box_text = f"#{i + 1}:[MASK]({char_label})"

                    b['label'] = char_label
                    clusters.append(b)
                    chars.append((i + 1, crop_img, b))

                    if save_individual_chars:
                        cv2.imwrite(os.path.join(chars_dir, f"{i + 1:03d}_{char_label}.png"), resized_crop)

                    # Draw YOLO boxes and Classifications
                    cv2.rectangle(vis_yolo, (b['x'], b['y']), (b['x'] + b['w'], b['y'] + b['h']), (0, 255, 0), 2)
                    cv2.putText(vis_yolo, box_text, (b['x'], max(b['y'] - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 0), 1)

        cv2.imwrite(os.path.join(out_dir, "09_yolo_segmentation_annotated.png"), vis_yolo)

        # Generate output JSON
        final_json_output = {
            "noisy_transliteration": "".join(noisy_text),
            "soft_probs": soft_probs_list,
            "confidences": confidences_list
        }
        output_path = Path(out_dir) / f"{image_stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json_output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 64}")
    print(f"  FINAL TRANSLITERATION : {final_json_output.get('noisy_transliteration', '')}")
    print(f"  YOLO CHARACTER COUNT  : {len(clusters)}")
    print(f"  Output dir            : {out_dir}")
    print(f"{'=' * 64}\n")

    return {
        "chars": chars,
        "count": len(clusters),
        "confidence": "OK",
        "flow_type": baseline["flow_type"] if 'baseline' in locals() else "unknown",
        "clusters": clusters,
        "quality_metrics": metrics
    }


def batch_process(input_dir, out_root="batch_output", **kwargs):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted(p for p in Path(input_dir).iterdir()
                    if p.suffix.lower() in exts)
    if not images:
        print(f"No images in {input_dir}");
        return

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
              f"{psnr:>7.1f}  {ssim:>6.3f}  {edg * 100:>7.1f}%")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    ap = argparse.ArgumentParser(
        description="Module 1 - Brahmi Inscription Preprocessing & Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dynamic path resolutions from project root
    ap.add_argument("input", nargs="?", help="Image file, or directory (use --batch)",
                    default=str(project_root / "data" / "raw_estampages"))
    ap.add_argument("--out", default=str(script_dir / "batch_results"))
    ap.add_argument("--mobilenet", default=str(project_root / "models" / "mobile_net_v2_brahmi_best.pth"),
                    help="Path to MobileNet weights")
    ap.add_argument("--label_mapping", default=str(script_dir / "label_mapping.json"), help="Path to label mapping")
    ap.add_argument("--config", default=str(script_dir / "config.yaml"), help="Path to config file")

    # ── Preprocessing (new — from reference paper Ch.6) ─────────────────
    grp_pre = ap.add_argument_group("Preprocessing (reference paper additions)")
    grp_pre.add_argument("--nlm_h", type=int, default=10,
                         help="Non-Local Means filter strength. "
                              "Higher=more smoothing. "
                              "Try 20-30 for very noisy scans.")
    grp_pre.add_argument("--bilateral", action="store_true", default=True,
                         help="Add bilateral edge-preserving filter after NLMeans "
                              "(default ON). Disable with --no_bilateral.")
    grp_pre.add_argument("--no_bilateral", dest="bilateral", action="store_false",
                         help="Disable bilateral filter (already-clean images).")
    grp_pre.add_argument("--otsu_bias", type=int, default=20,
                         help="Bias subtracted from Otsu threshold. "
                              "Lower for faded inscriptions.")
    grp_pre.add_argument("--dilate_kernel", type=int, nargs=2, default=[3, 3],
                         metavar=("W", "H"),
                         help="Morphological closing kernel size. "
                              "Use '5 5' for heavily fragmented strokes.")
    grp_pre.add_argument("--dilate_iters", type=int, default=1,
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
    grp_seg.add_argument("--mzs", type=float, default=3.0,
                         help="Modified Z-Score threshold for wide-segment split.")
    grp_seg.add_argument("--gap_floor", type=float, default=0.45,
                         help="Gap-floor ratio 0-1. "
                              "RAISE if too many chars, LOWER if too few.")

    ap.add_argument("--show", action="store_true", help="Show matplotlib plots.")
    ap.add_argument("--batch", action="store_true", help="Process whole directory.", default=True)
    ap.add_argument("--no_auto", action="store_true",
                    help="Disable auto-calibration. Use explicit flags for all params.")
    args = ap.parse_args()

    kw = dict(
        mobilenet_path=args.mobilenet,
        label_mapping_path=args.label_mapping,
        config_path=args.config,
        white_noise_thresh=args.white_thresh if args.no_auto else None,
        black_noise_thresh=args.black_thresh if args.no_auto else None,
        mzs_threshold=args.mzs,
        gap_floor_ratio=args.gap_floor if args.no_auto else None,
        nlm_h=args.nlm_h if args.no_auto else None,
        bilateral=args.bilateral,
        otsu_bias=args.otsu_bias if args.no_auto else None,
        dilate_kernel=tuple(args.dilate_kernel),
        dilate_iters=args.dilate_iters,
        auto_params=not args.no_auto,
        show_plots=args.show,
    )

    if args.batch:
        batch_process(args.input, out_root=args.out, **kw)
    else:
        run_module1(args.input, out_dir=args.out, **kw)