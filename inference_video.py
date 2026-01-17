"""
Inference on images OR videos with timestamp localization for videos.

Usage:
  python 4_inference.py path/to/image.jpg
  python 4_inference.py path/to/video.mp4

Outputs:
- Image: {input_type, is_fake, confidence}
- Video: {input_type, video_is_fake, overall_confidence, manipulated_segments:[{start_time,end_time,confidence}]}
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm


# ----------------------------
# Model (same as training)
# ----------------------------
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
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg"
        )
        self.freq_branch = FrequencyBranch()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
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
# Utils
# ----------------------------
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_bgr_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 -> normalized tensor (1,3,224,224)"""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return t


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def ema_smooth(scores, alpha=0.6):
    """Exponential moving average smoothing"""
    if not scores:
        return scores
    out = [scores[0]]
    for x in scores[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def scores_to_segments(
    scores,
    times,
    t_high=0.6,
    t_low=0.45,
    min_duration=0.5,
    merge_gap=0.35,
):
    """
    Hysteresis thresholding + merge segments.
    scores: list[float] fake probabilities (smoothed)
    times:  list[float] timestamp (seconds) for each score
    """
    segments = []
    in_seg = False
    start_t = None
    seg_scores = []

    for p, t in zip(scores, times):
        if not in_seg:
            if p >= t_high:
                in_seg = True
                start_t = t
                seg_scores = [p]
        else:
            seg_scores.append(p)
            if p <= t_low:
                end_t = t
                segments.append([start_t, end_t, float(np.mean(seg_scores))])
                in_seg = False
                start_t = None
                seg_scores = []

    # close open seg
    if in_seg and start_t is not None:
        segments.append([start_t, times[-1], float(np.mean(seg_scores))])

    # prune short
    segments = [s for s in segments if (s[1] - s[0]) >= min_duration]

    # merge close gaps
    if not segments:
        return []

    merged = [segments[0]]
    for s in segments[1:]:
        prev = merged[-1]
        gap = s[0] - prev[1]
        if gap <= merge_gap:
            # merge by extending end and averaging confidences weighted by duration
            prev_dur = max(prev[1] - prev[0], 1e-6)
            s_dur = max(s[1] - s[0], 1e-6)
            prev[1] = max(prev[1], s[1])
            prev[2] = float((prev[2] * prev_dur + s[2] * s_dur) / (prev_dur + s_dur))
        else:
            merged.append(s)

    return merged


# ----------------------------
# Predict image
# ----------------------------
def predict_image(model, image_path: str, device):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    x = preprocess_bgr_frame(img).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, 1)
        fake_prob = probs[0, 1].item()

    return {
        "input_type": "image",
        "is_fake": fake_prob > 0.5,
        "confidence": round(fake_prob, 4),
    }


# ----------------------------
# Predict video (with timestamps)
# ----------------------------
def predict_video(
    model,
    video_path: str,
    device,
    sample_fps=5,
    batch_size=16,
    smooth_alpha=0.6,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if not vid_fps or vid_fps <= 0:
        vid_fps = 30.0

    stride = max(int(round(vid_fps / sample_fps)), 1)

    frames = []
    times = []
    frame_idx = 0

    # read + sample
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            t = frame_idx / vid_fps
            frames.append(frame)
            times.append(t)
        frame_idx += 1

    cap.release()

    if not frames:
        return {
            "input_type": "video",
            "video_is_fake": False,
            "overall_confidence": 0.0,
            "manipulated_segments": [],
            "note": "No frames extracted from video."
        }

    # batch inference
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            xs = torch.cat([preprocess_bgr_frame(f) for f in batch], dim=0).to(device)
            outputs = model(xs)
            probs = torch.softmax(outputs, 1)[:, 1].detach().cpu().numpy().tolist()
            scores.extend(probs)

    # temporal smoothing
    smooth_scores = ema_smooth(scores, alpha=smooth_alpha)

    # overall confidence (mean of top-k helps partial fakes)
    k = max(1, int(0.1 * len(smooth_scores)))  # top 10%
    topk = sorted(smooth_scores, reverse=True)[:k]
    overall_conf = float(np.mean(topk))

    # decode segments
    segments = scores_to_segments(
        smooth_scores,
        times,
        t_high=0.6,
        t_low=0.45,
        min_duration=0.5,
        merge_gap=0.35,
    )

    manipulated_segments = [
        {
            "start_time": format_time(s[0]),
            "end_time": format_time(s[1]),
            "confidence": round(s[2], 4),
        }
        for s in segments
    ]

    return {
        "input_type": "video",
        "video_is_fake": overall_conf > 0.5,
        "overall_confidence": round(overall_conf, 4),
        "manipulated_segments": manipulated_segments,
    }


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python 4_inference.py <image_path>")
        print("  python 4_inference.py <video_path>")
        sys.exit(0)

    input_path = sys.argv[1]
    if not Path(input_path).exists():
        print(f"ERROR: path not found: {input_path}")
        sys.exit(1)

    device = get_device()
    print("Loading model...")
    model = DeepfakeDetector().to(device)

    ckpt_path = Path("models/best_model.pth")
    if not ckpt_path.exists():
        print("ERROR: models/best_model.pth not found. Train first or put weights there.")
        sys.exit(1)

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    auc = checkpoint.get("auc", None)
    if auc is not None:
        print(f"Model loaded (AUC: {auc:.4f})")
    else:
        print("Model loaded.")

    suffix = Path(input_path).suffix.lower()
    if suffix in [".jpg", ".jpeg", ".png"]:
        result = predict_image(model, input_path, device)
    else:
        result = predict_video(model, input_path, device)

    print("\nResult:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
