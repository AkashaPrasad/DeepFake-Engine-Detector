"""
Train deepfake detection model (image-level) from extracted frames / face crops.

Expected folder structure (recommended):
data/processed/
  real/   (contains images, any nesting)
  fake/   (contains images, any nesting)

Example:
data/processed/real/video_0001/frame_0001.jpg
data/processed/fake/video_0001/frame_0001.jpg
"""

import os
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# ----------------------------
# Config
# ----------------------------
DATA_ROOT = Path("data/processed")
REAL_DIR = DATA_ROOT / "real"
FAKE_DIR = DATA_ROOT / "fake"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

os.makedirs("models", exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Model
# ----------------------------
class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # FFT on grayscale
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray.unsqueeze(1))
        magnitude = torch.abs(torch.fft.fftshift(fft))
        magnitude = torch.log(magnitude + 1e-8)
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)

        x = magnitude.repeat(1, 3, 1, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.pool(x).flatten(1)  # (B, 64)


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # timm create_model supports num_classes=0 to return features instead of logits :contentReference[oaicite:1]{index=1}
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.freq_branch = FrequencyBranch()

        # infer backbone dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            backbone_dim = self.backbone(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        rgb_feat = self.backbone(x)
        freq_feat = self.freq_branch(x)
        combined = torch.cat([rgb_feat, freq_feat], dim=1)
        return self.classifier(combined)


# ----------------------------
# Dataset
# ----------------------------
class SimpleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _read_image(self, img_path: str):
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = self._read_image(img_path)
        if img is None:
            # If OpenCV can't read (bad path / corrupt file), return a dummy black image
            # and keep training instead of crashing.
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        # Simple augmentation
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        # Normalize to ImageNet stats
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return img, torch.tensor(label, dtype=torch.long)


# ----------------------------
# Data loading helpers
# ----------------------------
def collect_images(folder: Path, label: int):
    """Recursively collect images under folder."""
    samples = []
    if not folder.exists():
        return samples
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            samples.append((str(p), label))
    return samples


def safe_train_val_split(all_samples, test_size=0.2):
    labels = [s[1] for s in all_samples]
    n_total = len(all_samples)
    n_real = sum(1 for y in labels if y == 0)
    n_fake = sum(1 for y in labels if y == 1)

    # If one class is too small, stratify will error; do a plain split instead.
    # sklearn stratify requires enough samples per class :contentReference[oaicite:2]{index=2}
    can_stratify = (n_real >= 2 and n_fake >= 2 and n_total >= 5)

    if n_total == 0:
        raise ValueError("No samples found.")

    if can_stratify:
        return train_test_split(
            all_samples,
            test_size=test_size,
            random_state=SEED,
            stratify=labels,
        )
    else:
        return train_test_split(
            all_samples,
            test_size=test_size,
            random_state=SEED,
            shuffle=True,
        )


# ----------------------------
# Training
# ----------------------------
def main():
    set_seed(SEED)

    print("Loading dataset...")
    print(f"Looking for images under:\n  REAL: {REAL_DIR}\n  FAKE: {FAKE_DIR}")

    all_samples = []
    all_samples += collect_images(REAL_DIR, 0)
    all_samples += collect_images(FAKE_DIR, 1)

    print(f"Found samples: {len(all_samples)}")
    if len(all_samples) == 0:
        print("\nERROR: Total samples = 0\n")
        print("Fix your folder structure first. Your script expects images (jpg/png) here:")
        print("  data/processed/real/**.jpg (or .png)")
        print("  data/processed/fake/**.jpg (or .png)")
        print("\nIf you only have videos right now, you must extract frames/faces first.")
        print("Example structure:")
        print("  data/processed/real/video_0001/frame_0001.jpg")
        print("  data/processed/fake/video_0002/frame_0001.jpg")
        return

    # Split
    try:
        train_samples, val_samples = safe_train_val_split(all_samples, test_size=0.2)
    except Exception as e:
        print(f"Split failed: {e}")
        return

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_dataset = SimpleDataset(train_samples)
    val_dataset = SimpleDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = get_device()
    print(f"Using device: {device}")

    model = DeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_auc = 0.0

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        # ---- Train
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping helps stability (especially on small datasets)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, 1).detach().cpu().numpy().tolist())
            train_labels.extend(labels.detach().cpu().numpy().tolist())

        train_acc = accuracy_score(train_labels, train_preds)

        # ---- Validate
        model.eval()
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, 1)

                val_preds.extend(torch.argmax(outputs, 1).cpu().numpy().tolist())
                val_labels.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs[:, 1].cpu().numpy().tolist())

        val_acc = accuracy_score(val_labels, val_preds)

        # roc_auc_score requires both classes present in val
        if len(set(val_labels)) < 2:
            val_auc = float("nan")
        else:
            val_auc = roc_auc_score(val_labels, val_probs)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc}")

        # Save best model (only if AUC is valid)
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "auc": float(val_auc),
                },
                "models/best_model.pth",
            )
            print(f"✓ Saved best model (AUC: {val_auc:.4f})")

    print(f"\nTraining complete! Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
