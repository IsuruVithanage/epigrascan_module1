"""
Step 4: Train Brahmi Shape Classifier (MobileNetV2)
With curriculum augmentation for synthetic noise
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
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


def get_transforms(config, is_training=True):
    if is_training:
        return A.Compose([
            A.Resize(config["model"]["image_size"], config["model"]["image_size"]),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            # Curriculum noise will be added later in training loop
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(config["model"]["image_size"], config["model"]["image_size"]),
            ToTensorV2()
        ])


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load dataset (Kaggle clean + synthetic)
    train_dataset = ImageFolder(
        root=config["paths"]["kaggle_brahmi"],
        transform=lambda x: get_transforms(config, True)(image=np.array(x))['image']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 2. Initialize model
    model = BrahmiClassifier(num_classes=config["model"]["num_classes"]).to(device)

    # 3. Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # 4. Training loop
    best_acc = 0.0
    for epoch in range(config["training"]["epochs"]):
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