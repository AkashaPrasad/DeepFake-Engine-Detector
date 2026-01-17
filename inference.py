"""
Test model on images or videos
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import json
import sys

# Load model (copy from 3_train.py)
import torch.nn as nn
import timm

class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray.unsqueeze(1))
        magnitude = torch.abs(torch.fft.fftshift(fft))
        magnitude = torch.log(magnitude + 1e-8)
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
        x = magnitude.repeat(1, 3, 1, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.pool(x).flatten(1)

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')
        self.freq_branch = FrequencyBranch()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        rgb_feat = self.backbone(x)
        freq_feat = self.freq_branch(x)
        combined = torch.cat([rgb_feat, freq_feat], dim=1)
        return self.classifier(combined)

def predict_image(model, image_path, device):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, 1)
        fake_prob = probs[0, 1].item()
    
    return {
        "input_type": "image",
        "is_fake": fake_prob > 0.5,
        "confidence": round(fake_prob, 4)
    }

# Load model
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

print("Loading model...")
device = get_device()
model = DeepfakeDetector().to(device)

checkpoint = torch.load('models/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded (AUC: {checkpoint['auc']:.4f})")

# Test
if len(sys.argv) < 2:
    print("\nUsage: python 4_inference.py <image_path>")
    print("Example: python 4_inference.py test_image.jpg")
else:
    image_path = sys.argv[1]
    result = predict_image(model, image_path, device)
    print("\nResult:")
    print(json.dumps(result, indent=2))
