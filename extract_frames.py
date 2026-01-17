"""
Extract frames from videos in:
data/raw/real/*.mp4
data/raw/fake/*.mp4

Save to:
data/processed/real/<video_name>/*.jpg
data/processed/fake/<video_name>/*.jpg
"""

import cv2
from pathlib import Path

# ================= CONFIG =================
RAW_ROOT = Path("data/raw")
OUT_ROOT = Path("data/processed")

FPS = 5          # frames per second
MAX_FRAMES = 50  # max frames per video
IMG_SIZE = 224
# ==========================================


def extract_from_dir(video_dir: Path, label: str):
    print(f"\n🔍 Scanning: {video_dir.resolve()}")

    if not video_dir.exists():
        print(f"❌ Folder not found: {video_dir}")
        return

    out_base = OUT_ROOT / label
    out_base.mkdir(parents=True, exist_ok=True)

    videos = list(video_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos")

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open {video_path.name}")
            continue

        video_name = video_path.stem
        out_dir = out_base / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(int(video_fps // FPS), 1)

        frame_id = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                out_path = out_dir / f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1

            frame_id += 1
            if saved >= MAX_FRAMES:
                break

        cap.release()
        print(f"✅ {label.upper()} | {video_name}: {saved} frames")


def main():
    print("Starting frame extraction...")

    extract_from_dir(RAW_ROOT / "real", "real")
    extract_from_dir(RAW_ROOT / "fake", "fake")

    print("\n🎉 Frame extraction complete!")


if __name__ == "__main__":
    main()
