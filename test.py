import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Load your custom trained model
model = YOLO('best.pt')

input_folder = "data/testimages"
output_folder = "data/grid_outputs"
os.makedirs(output_folder, exist_ok=True)

print(f"Generating optimized grids for images in: {input_folder}")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # TWEAK 1: Lower confidence to catch missing chars, adjust IoU to allow overlapping boxes
        # This will bring your character count back up to ~26
        results = model(img, conf=0.25, iou=0.6)

        raw_boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(raw_boxes) == 0:
            print(f"No characters found in {filename}, skipping.")
            continue

        processed_boxes = []
        for box in raw_boxes:
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)

            # TWEAK 2: Relaxed geometric filters
            # We removed the aspect ratio constraint entirely to stop killing valid characters.
            # We only filter out boxes smaller than 8x8 pixels (pure noise/dust).
            if w > 8 and h > 8:
                processed_boxes.append({'box': (x1, y1, x2, y2), 'cx': cx, 'cy': cy, 'h': h})

        if not processed_boxes:
            continue

        # TWEAK 3: Smart Row Clustering (Fixes the Reading Order)
        # Step A: Sort all boxes by their Y-center first (top of page to bottom)
        processed_boxes.sort(key=lambda b: b['cy'])

        median_height = np.median([b['h'] for b in processed_boxes])
        row_tolerance = median_height * 0.4  # 40% of a character's height variance allowed

        rows = []
        current_row = [processed_boxes[0]]

        # Step B: Group boxes into distinct horizontal lines
        for box in processed_boxes[1:]:
            current_row_cy = sum(b['cy'] for b in current_row) / len(current_row)

            # If the box is vertically aligned with the current row, add it
            if abs(box['cy'] - current_row_cy) < row_tolerance:
                current_row.append(box)
            else:
                # Otherwise, it belongs to the next line of text down
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)  # Append the final row

        # Step C: Sort each individual row from left-to-right (by X-center)
        for row in rows:
            row.sort(key=lambda b: b['cx'])

        # Flatten the list back out into a single ordered sequence
        final_sorted_boxes = [box for row in rows for box in row]

        # Extract the images based on our new perfect order
        crops = []
        for item in final_sorted_boxes:
            x1, y1, x2, y2 = item['box']
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        total_chars = len(crops)

        # Set up the Matplotlib grid
        cols = 10
        rows_grid = math.ceil(total_chars / cols)

        fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 2, rows_grid * 2))
        fig.suptitle(f'Segmented Characters from {filename} (total = {total_chars})', fontsize=16)

        if rows_grid * cols == 1:
            axes_flat = [axes]
        elif rows_grid == 1 or cols == 1:
            axes_flat = axes
        else:
            axes_flat = axes.flatten()

        for i, ax in enumerate(axes_flat):
            if i < total_chars:
                ax.imshow(crops[i])
                ax.set_title(f'#{i + 1}', fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        output_grid_path = os.path.join(output_folder, f"grid_{filename}")
        plt.savefig(output_grid_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Grid saved: {output_grid_path} | Valid chars: {total_chars}")

print(f"All grids successfully generated in: {output_folder}")