"""
Step 4: Train Brahmi Shape Classifier (MobileNetV2)
With curriculum augmentation for synthetic noise
"""
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

class BrahmiClassifier(nn.Module):
    def __init__(self, num_classes=368):
        super().__init__()
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)


def get_transforms(config, is_training=True, heavy_noise=False):
    # Standard ImageNet normalization values
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_training:
        # Base augmentations (always applied)
        aug_list = [
            A.Resize(config["model"]["image_size"], config["model"]["image_size"]),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ]

        # --- THE FIX: Curriculum Heavy Noise ---
        if heavy_noise:
            aug_list.extend([
                # Simulates stone grain / salt-and-pepper noise
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                # Simulates large eroded patches, cracks, or missing chunks of rock
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),
                # Simulates warped/distorted strokes from bad estampage rubbing
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3)
            ])

        aug_list.extend([normalize, ToTensorV2()])
        return A.Compose(aug_list)
    else:
        return A.Compose([
            A.Resize(config["model"]["image_size"], config["model"]["image_size"]),
            normalize,
            ToTensorV2()
        ])


def morphological_sandwich(img_path):
    """Loads an image and applies the exact same preprocessing used in inference."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
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

    # Convert back to 3-channel RGB so MobileNetV2 accepts it
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    return cleaned_rgb

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure model directory exists
    Path(config["paths"]["models"]).mkdir(parents=True, exist_ok=True)

    # 1. Load dataset (Kaggle clean + synthetic)
    train_transform = get_transforms(config, True)

    train_dataset = ImageFolder(
        root=config["paths"]["kaggle_brahmi"],
        loader=morphological_sandwich,
        transform=lambda x: train_transform(image=x)['image']
        # No need for np.array(x) anymore because OpenCV returns numpy arrays!
    )

    # --- THE FIX: Dynamically determine the number of classes ---
    actual_num_classes = len(train_dataset.classes)
    print(f"Detected {actual_num_classes} classes in the dataset.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 2. Initialize model using the dynamic class count
    model = BrahmiClassifier(num_classes=actual_num_classes).to(device)

    # 3. Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # 4. Training loop
    best_acc = 0.0
    for epoch in range(config["training"]["epochs"]):

        if epoch == config["training"].get("curriculum_start_epoch", 6):
            print("\n🌪️ [Curriculum Learning] Activating Heavy Synthetic Noise for Erosion/Cracks!")
            heavy_transform = get_transforms(config, is_training=True, heavy_noise=True)
            # Hot-swap the transform on the existing dataset
            train_dataset.transform = lambda x: heavy_transform(image=np.array(x))['image']

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{running_loss / len(train_loader):.4f}",
                             acc=f"{100. * correct / total:.2f}%")

        scheduler.step()

        # Save best model
        current_acc = correct / total
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(),
                       Path(config["paths"]["models"]) / "mobile_net_v2_brahmi_best.pth")
            print(f"   New best model saved! Accuracy: {best_acc * 100:.2f}%")

    print("\n✅ Training completed!")
    print(f"Best training accuracy: {best_acc * 100:.2f}%")
    print(f"Model saved as: models/mobile_net_v2_brahmi_best.pth")

if __name__ == "__main__":
    main()