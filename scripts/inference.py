"""
Final Robust Inference Script
"""

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import json
from pathlib import Path
import yaml
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

class BrahmiClassifier(nn.Module):
    def __init__(self, num_classes=368):
        super().__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)

def morphological_sandwich(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    cleaned = cv2.medianBlur(dilated, 3)
    return cleaned


def isolate_characters(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_crops = []
    positions = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        crop = binary[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (224, 224))
        char_crops.append(crop)
        positions.append(x)

    # THE FIX IS HERE:
    return [crop for _, crop in sorted(zip(positions, char_crops), key=lambda item: item[0])]

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label mapping to get the exact number of classes
    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    actual_num_classes = len(label_mapping)

    model = BrahmiClassifier(num_classes=actual_num_classes).to(device)
    model.load_state_dict(torch.load("models/mobile_net_v2_brahmi_best.pth", map_location=device))
    model.eval()
    print(f"✅ Loaded model successfully.")

    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    raw_dir = Path(config["paths"]["raw_estampages"])
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(exist_ok=True)

    image_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg"))

    print(f"Found {len(image_files)} raw estampages...\n")

    for img_path in tqdm(image_files, desc="Generating JSON for CAME"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            binary = morphological_sandwich(img)
            chars = isolate_characters(binary)

            noisy_text = []
            soft_probs_list = []
            confidences_list = []

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            for char_crop in chars:
                char_rgb = cv2.cvtColor(char_crop, cv2.COLOR_GRAY2RGB)
                tensor = transform(char_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    confidence = float(probs.max())

                if confidence >= config["model"]["confidence_threshold"]:
                    pred_idx = int(probs.argmax())
                    char_label = label_mapping.get(str(pred_idx), "?")
                    noisy_text.append(char_label)
                else:
                    noisy_text.append("[MASK]")
                    soft_probs_list.append(probs.tolist())
                    confidences_list.append(confidence)

            result = {
                "noisy_transliteration": "".join(noisy_text),
                "soft_probs": soft_probs_list,
                "confidences": confidences_list
            }

            output_path = output_dir / f"{img_path.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"❌ Skipped {img_path.name} → {e}")

    print("\n🎉 All done!")
    print(f"✅ JSON files generated in: {output_dir}")

if __name__ == "__main__":
    main()