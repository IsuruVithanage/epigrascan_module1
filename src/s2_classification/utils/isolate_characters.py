"""
Step 3: Top-Down Line-First Character Isolation (Safe-Mode Edition)
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def merge_bounding_boxes(boxes):
    if not boxes: return []

    avg_w = np.mean([b[2] for b in boxes])
    avg_h = np.mean([b[3] for b in boxes])

    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip_indices = set()

        for i in range(len(boxes)):
            if i in skip_indices: continue

            x1, y1, w1, h1 = boxes[i]
            r1, b1 = x1 + w1, y1 + h1

            merged_this_round = False
            for j in range(i + 1, len(boxes)):
                if j in skip_indices: continue

                x2, y2, w2, h2 = boxes[j]
                r2, b2 = x2 + w2, y2 + h2

                x_overlap = max(0, min(r1, r2) - max(x1, x2))
                y_overlap = max(0, min(b1, b2) - max(y1, y2))

                x_overlap_ratio = x_overlap / min(w1, w2) if min(w1, w2) > 0 else 0
                y_overlap_ratio = y_overlap / min(h1, h2) if min(h1, h2) > 0 else 0

                gap_x = max(0, max(x1, x2) - min(r1, r2))
                gap_y = max(0, max(y1, y2) - min(b1, b2))

                area1, area2 = w1*h1, w2*h2
                intersection = x_overlap * y_overlap
                containment_ratio = intersection / min(area1, area2) if min(area1, area2) > 0 else 0

                should_merge = False

                if x_overlap_ratio >= 0.50 and gap_y <= (0.20 * avg_h):
                    should_merge = True
                elif y_overlap_ratio >= 0.40 and gap_x <= (0.05 * avg_w):
                    should_merge = True
                elif containment_ratio >= 0.80:
                    should_merge = True

                if should_merge:
                    nx, ny = min(x1, x2), min(y1, y2)
                    nw, nh = max(r1, r2) - nx, max(b1, b2) - ny

                    if nw > (avg_w * 2.5):
                        should_merge = False
                    else:
                        new_boxes.append((nx, ny, nw, nh))
                        skip_indices.add(i)
                        skip_indices.add(j)
                        merged = True
                        merged_this_round = True
                        break

            if not merged_this_round:
                new_boxes.append(boxes[i])

        boxes = new_boxes
    return boxes

def isolate_characters(image_path: str, config: dict):
    try:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None: raise ValueError("Could not read image")
        img_h, img_w = gray.shape

        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(binary) > (img_h * img_w * 0.5):
            binary = cv2.bitwise_not(binary)

        binary[0:5, :] = 0; binary[-5:, :] = 0; binary[:, 0:5] = 0; binary[:, -5:] = 0
        preview_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # ==========================================================
        # STAGE 1: ZERO-BLEED SMEARING
        # (60, 1) dilates left-to-right heavily, but absolutely ZERO up-and-down.
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        smeared_lines = cv2.dilate(binary, line_kernel, iterations=2)

        # Save X-Ray of the Smear so we can visually debug it
        debug_dir = Path("data/debug_bounds")
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{Path(image_path).stem}_smear_xray.jpg"), smeared_lines)

        line_contours, _ = cv2.findContours(smeared_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_lines = []
        for l_cnt in line_contours:
            lx, ly, lw, lh = cv2.boundingRect(l_cnt)
            if lh < 10 or lw < 20: continue # Only kill tiny dust, let giant lines survive!
            valid_lines.append((lx, ly, lw, lh))

        valid_lines.sort(key=lambda b: b[1])

        if not valid_lines:
            print(f"\n  [!] {Path(image_path).name}: Found 0 text lines during smearing.")
            return []
        # ==========================================================

        target_size = config["model"]["image_size"]
        final_sorted_crops = []

        # STAGE 2: PER-LINE ISOLATION
        for line_idx, (lx, ly, lw, lh) in enumerate(valid_lines):
            line_roi = binary[ly:ly+lh, lx:lx+lw]

            # Gentle Snapper: (3,3) protects thin letters!
            snap_kernel = np.ones((3, 3), np.uint8)
            snapped_roi = cv2.morphologyEx(line_roi, cv2.MORPH_OPEN, snap_kernel)

            char_contours, _ = cv2.findContours(snapped_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            raw_boxes = []
            for c_cnt in char_contours:
                cx, cy, cw, ch = cv2.boundingRect(c_cnt)
                area = cv2.contourArea(c_cnt)

                # Relaxed the Ghost Killer from 0.15 to 0.08 so crossed-letters survive
                if area < 20 or (area / (cw * ch) < 0.08): continue
                raw_boxes.append((cx, cy, cw, ch))

            if not raw_boxes:
                print(f"  [!] {Path(image_path).name}: Line {line_idx+1} dropped (0 characters survived filters).")
                continue

            merged_boxes = merge_bounding_boxes(raw_boxes)

            line_boxes = []
            widths = [b[2] for b in merged_boxes]
            median_w = np.median(widths)
            mad = np.median([abs(w - median_w) for w in widths])
            if mad == 0: mad = 1.0

            for (cx, cy, cw, ch) in merged_boxes:
                z_score = 0.6745 * (cw - median_w) / mad
                if z_score > 3.0:
                    num_letters = max(2, round(cw / median_w))
                    slice_w = cw // num_letters
                    for i in range(num_letters):
                        line_boxes.append((cx + (i * slice_w), cy, slice_w, ch))
                else:
                    line_boxes.append((cx, cy, cw, ch))

            line_boxes.sort(key=lambda b: b[0])
            centroids = []

            for (cx, cy, cw, ch) in line_boxes:
                global_x = lx + cx
                global_y = ly + cy

                cv2.rectangle(preview_img, (global_x, global_y), (global_x + cw, global_y + ch), (0, 255, 0), 2)

                center_x = global_x + (cw // 2)
                center_y = global_y + (ch // 2)
                centroids.append((center_x, center_y))
                cv2.circle(preview_img, (center_x, center_y), 4, (0, 0, 255), -1)

                crop = binary[global_y:global_y + ch, global_x:global_x + cw]
                if crop.size == 0: continue

                margin = int(max(ch, cw) * 0.15)
                diff = abs(ch - cw)
                pad_top, pad_bottom, pad_left, pad_right = margin, margin, margin, margin

                if ch > cw:
                    pad_left += diff // 2
                    pad_right += (diff - diff // 2)
                elif cw > ch:
                    pad_top += diff // 2
                    pad_bottom += (diff - diff // 2)

                square_crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                                 cv2.BORDER_CONSTANT, value=0)
                final_crop = cv2.resize(square_crop, (target_size, target_size))
                final_sorted_crops.append(final_crop)

            if len(centroids) > 3:
                centroids_np = np.array(centroids)
                x_points = centroids_np[:, 0]
                y_points = centroids_np[:, 1]

                _, unique_indices = np.unique(x_points, return_index=True)
                unique_indices.sort()

                if len(unique_indices) > 3:
                    x_points = x_points[unique_indices]
                    y_points = y_points[unique_indices]

                    x_smooth = np.linspace(x_points.min(), x_points.max(), 300)
                    spl = make_interp_spline(x_points, y_points, k=3)
                    y_smooth = spl(x_smooth)

                    curve_points = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
                    cv2.polylines(preview_img, [curve_points], isClosed=False, color=(255, 0, 0), thickness=2)

        cv2.imwrite(str(debug_dir / f"{Path(image_path).stem}_bounds.jpg"), preview_img)

        return final_sorted_crops

    except Exception as e:
        print(f"  Error processing {Path(image_path).name}: {e}")
        return []

def main():
    config = load_config()
    isolated_dir = Path("data/isolated")
    isolated_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(config["paths"]["processed"])

    image_files = list(processed_dir.glob("*.[jp][pn]*"))
    print(f"Found {len(image_files)} cleaned images...\n")

    total_chars = 0
    for img_path in tqdm(image_files, desc="Isolating characters"):
        chars = isolate_characters(str(img_path), config)
        base_name = img_path.stem

        for i, char_img in enumerate(chars):
            output_path = isolated_dir / f"{base_name}_char_{i:03d}.png"
            cv2.imwrite(str(output_path), char_img)
        total_chars += len(chars)

    print(f"\n✅ Isolation completed! Saved {total_chars} characters.")

if __name__ == "__main__":
    main()