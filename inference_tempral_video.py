"""
Video inference with temporal localisation (timestamp segments).

Usage:
  python inference_video_temporal.py /path/to/video.mp4
  python inference_video_temporal.py /path/to/video.mp4 models/best_model.pth

Outputs JSON:
{
  "input_type": "video",
  "video_path": "...",
  "video_is_fake": true/false,
  "overall_confidence": float,
  "sample_fps": int,
  "threshold_used": float,
  "manipulated_segments": [
      {"start_time":"HH:MM:SS.xx","end_time":"HH:MM:SS.xx","confidence": float}
  ]
}

Notes:
- Uses your existing frame model (EfficientNet-B0 + FFT branch).
- Does NOT require retraining.
- Temporal localisation is derived from frame scores using smoothing + hysteresis.
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm


# ---------------- CONFIG (safe defaults) ----------------
IMAGE_SIZE = 224
SAMPLE_FPS = 5           # sample 5 frames per second
BATCH_SIZE = 16
SMOOTH_ALPHA = 0.6       # EMA smoothing strength

# Hysteresis thresholds for segments
T_HIGH = 0.60            # enter fake segment if score >= T_HIGH
T_LOW = 0.45             # exit fake segment if score <= T_LOW

MIN_SEGMENT_SEC = 0.50   # discard segments shorter than this
MERGE_GAP_SEC = 0.35     # merge segments if gap between them <= this

# Overall video decision threshold (after aggregation)
VIDEO_THRESHOLD = 0.50
# --------------------------------------------------------


# ---------------- MODEL (same as training) ----------------
class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray.unsqueeze(1))
        mag = torch.abs(torch.fft.fftshift(fft))
        mag = torch.log(mag + 1e-8)
        mag = (mag - mag.mean()) / (mag.std() + 1e-8)

        x = mag.repeat(1, 3, 1, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.pool(x).flatten(1)


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.freq_branch = FrequencyBranch()

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
        return self.classifier(torch.cat([rgb_feat, freq_feat], dim=1))
# ----------------------------------------------------------


# ---------------- UTILS ----------------
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_bgr(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 -> normalized tensor (1,3,224,224)"""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
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
    if not scores:
        return scores
    out = [scores[0]]
    for x in scores[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def scores_to_segments(scores, times,
                       t_high=T_HIGH, t_low=T_LOW,
                       min_duration=MIN_SEGMENT_SEC,
                       merge_gap=MERGE_GAP_SEC):
    """
    Convert smoothed scores to temporal segments using hysteresis thresholding.
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
                conf = float(np.mean(seg_scores)) if seg_scores else float(p)
                segments.append([start_t, end_t, conf])
                in_seg = False
                start_t = None
                seg_scores = []

    if in_seg and start_t is not None:
        end_t = times[-1]
        conf = float(np.mean(seg_scores)) if seg_scores else float(scores[-1])
        segments.append([start_t, end_t, conf])

    # drop tiny segments
    segments = [s for s in segments if (s[1] - s[0]) >= min_duration]

    # merge close segments
    if not segments:
        return []

    merged = [segments[0]]
    for s in segments[1:]:
        prev = merged[-1]
        gap = s[0] - prev[1]
        if gap <= merge_gap:
            prev_dur = max(prev[1] - prev[0], 1e-6)
            s_dur = max(s[1] - s[0], 1e-6)
            prev[1] = max(prev[1], s[1])
            prev[2] = float((prev[2] * prev_dur + s[2] * s_dur) / (prev_dur + s_dur))
        else:
            merged.append(s)

    return merged


def aggregate_overall(scores):
    """
    Robust overall confidence:
    mean of top 10% scores (helps partial manipulations).
    """
    if not scores:
        return 0.0
    k = max(1, int(0.1 * len(scores)))
    topk = sorted(scores, reverse=True)[:k]
    return float(np.mean(topk))
# ----------------------------------------------------------


def predict_video(model, video_path: str, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if not vid_fps or vid_fps <= 0:
        vid_fps = 30.0

    stride = max(int(round(vid_fps / SAMPLE_FPS)), 1)

    frames = []
    times = []
    frame_idx = 0

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
            "video_path": video_path,
            "video_is_fake": False,
            "overall_confidence": 0.0,
            "sample_fps": SAMPLE_FPS,
            "threshold_used": VIDEO_THRESHOLD,
            "manipulated_segments": [],
            "note": "No frames extracted from video."
        }

    # Batch inference
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i:i + BATCH_SIZE]
            xs = torch.cat([preprocess_bgr(f) for f in batch], dim=0).to(device)
            outputs = model(xs)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy().tolist()
            scores.extend(probs)

    # Smooth + segments
    smooth_scores = ema_smooth(scores, alpha=SMOOTH_ALPHA)
    overall_conf = aggregate_overall(smooth_scores)
    segments = scores_to_segments(smooth_scores, times)

    result_segments = [
        {
            "start_time": format_time(s[0]),
            "end_time": format_time(s[1]),
            "confidence": round(float(s[2]), 4)
        }
        for s in segments
    ]

    return {
        "input_type": "video",
        "video_path": video_path,
        "video_is_fake": overall_conf >= VIDEO_THRESHOLD,
        "overall_confidence": round(float(overall_conf), 4),
        "sample_fps": SAMPLE_FPS,
        "threshold_used": VIDEO_THRESHOLD,
        "manipulated_segments": result_segments
    }


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inference_video_temporal.py /path/to/video.mp4")
        print("  python inference_video_temporal.py /path/to/video.mp4 models/best_model.pth")
        sys.exit(0)

    video_path = sys.argv[1]
    ckpt_path = sys.argv[2] if len(sys.argv) >= 3 else "models/best_model_celeb.pth"

    if not Path(video_path).exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    if not Path(ckpt_path).exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")
    print("Loading model...")

    model = DeepfakeDetector().to(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Running video inference with temporal localisation...")
    result = predict_video(model, video_path, device)

    print("\nResult:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()