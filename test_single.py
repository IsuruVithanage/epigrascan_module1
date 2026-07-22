import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import json
import sys
import numpy as np
import yaml


# 1. Model architecture (unchanged)
class BrahmiClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)


# 2. FIXED sandwich: pads with white margin BEFORE CLAHE/adaptiveThreshold so a
#    tight crop's own edges don't get misread as false ink, and KEEPS that
#    margin in the final image (does not crop it back off).
#
#    Tested empirically: keeping the margin in the 224x224 canvas raised
#    top-1 confidence on image1 from 10.8% -> 37.2%. Cropping the margin back
#    off afterward (so the glyph fills the frame edge-to-edge like the
#    original buggy version) UNDOES the fix (confidence drops back to ~11%).
#    This strongly suggests your Kaggle training glyphs have natural white
#    margin around the character, and the model expects that framing - a
#    tightly-cropped, edge-to-edge character (like your Stage-1 segmenter
#    currently produces with only 4px padding) is itself out of distribution.
#
#    IMPORTANT: the deployed MODEL was trained with the OLD (unpadded, no
#    margin) sandwich. Using this corrected function at inference only, with
#    no retraining, is still a train/test mismatch - just a smaller one. Use
#    this script to confirm the diagnosis and get better manual-test numbers;
#    for a real fix, apply the same change to train_classifier.py's
#    morphological_sandwich() and RETRAIN (see notes at the bottom).
def morphological_sandwich(img, pad_frac=0.6):
    h, w = img.shape[:2]
    pad = max(15, int(pad_frac * max(h, w)))
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))

    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    cleaned = cv2.medianBlur(dilated, 3)

    # NOTE: the margin is intentionally KEPT (not cropped back off).
    # Cropping it back off was tested and reproduces the original bug.
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)


def test_image(image_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        conf_threshold = config["model"]["confidence_threshold"]
    except Exception:
        conf_threshold = 0.40   # fallback if config.yaml isn't found

    model = BrahmiClassifier(num_classes=len(label_mapping)).to(device)
    model.load_state_dict(torch.load("models/mobile_net_v2_brahmi_best.pth", map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    processed_img = morphological_sandwich(img)
    img_resized = cv2.resize(processed_img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top5_idx = probs.argsort()[::-1][:5]
    pred_idx = top5_idx[0]
    confidence = probs[pred_idx]
    predicted_class = label_mapping.get(str(pred_idx), "?")

    print("\n" + "=" * 48)
    print(f" Image: {image_path}")
    if confidence >= conf_threshold:
        print(f" Predicted Class: {predicted_class}")
    else:
        print(f" Predicted Class: ? (best guess was '{predicted_class}', "
              f"but below {conf_threshold*100:.0f}% confidence threshold)")
    print(f" Confidence: {confidence * 100:.2f}%")
    print(" Top-5:")
    for i in top5_idx:
        print(f"   {label_mapping.get(str(i), '?'):>6}  {probs[i]*100:5.2f}%")
    print("=" * 48 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single.py <path_to_image>")
    else:
        test_image(sys.argv[1])

# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# 1. This padded sandwich fixes the border-artifact bug for tight crops, but
#    the MODEL was trained with the OLD (unpadded, no-margin) sandwich. Using
#    this new function at inference only, without retraining, is still a
#    train/test mismatch (smaller than before, but real). Treat this script
#    as a diagnostic tool, not a permanent fix, until you retrain.
#
# 2. To make this the real fix:
#      a) Apply the identical "pad with margin, keep the margin" change
#         inside train_classifier.py's morphological_sandwich().
#      b) Also raise crop_characters()'s padding in
#         src/s1_segmentation/modules/segmentation.py from 4px to something
#         proportionally similar to pad_frac above, so production character
#         crops carry the same margin the classifier is trained to expect.
#      c) Retrain the classifier so it learns from the corrected, padded
#         preprocessing end-to-end.
#
# 3. Even after that fix, low-confidence out-of-distribution inputs (like
#    image 2 in this diagnosis, which never exceeded ~7% confidence at any
#    padding) are a SEPARATE problem: the input glyph's style/resolution
#    likely differs from your Kaggle training images. Check stroke width,
#    resolution, and rendering style (anti-aliased vs hard-edge) of a sample
#    of your actual training images against your real segmented crops.
# ---------------------------------------------------------------------------