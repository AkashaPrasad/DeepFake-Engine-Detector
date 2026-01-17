"""
Extract faces from videos
"""
import cv2
import torch
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
import json

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

print("Loading face detector...")
device = get_device()
detector = MTCNN(keep_all=False, device=device, min_face_size=80)

def extract_faces_from_video(video_path, output_dir, label):
    """Extract face crops from video"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_idx = 0
    saved_count = 0
    
    while cap.isOpened() and frame_idx < 300:  # Max 300 frames per video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every 5th frame
        if frame_idx % 5 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = detector.detect(rgb)
            
            if boxes is not None and len(boxes) > 0:
                # Get face with highest confidence
                best_idx = 0
                if len(boxes) > 1:
                    best_idx = probs.argmax()
                
                if probs[best_idx] > 0.9:  # High confidence only
                    x1, y1, x2, y2 = boxes[best_idx].astype(int)
                    
                    # Add margin
                    h, w = frame.shape[:2]
                    margin = int((x2 - x1) * 0.3)
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)
                    
                    # Extract and resize face
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face = cv2.resize(face, (224, 224))
                        filename = f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(output_dir / filename), face)
                        saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    
    # Save metadata
    metadata = {
        'video_path': str(video_path),
        'label': label,
        'frames_extracted': saved_count,
        'fps': fps
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return saved_count

# Process all videos
print("\nProcessing videos...")
data_root = Path("data/raw")
output_root = Path("data/processed")

# Process real videos
real_videos = list((data_root / "real").glob("*.mp4")) + list((data_root / "real").glob("*.avi"))
print(f"\nFound {len(real_videos)} real videos")

for video_path in tqdm(real_videos, desc="Real videos"):
    output_dir = output_root / "real" / video_path.stem
    if not (output_dir / "metadata.json").exists():
        try:
            count = extract_faces_from_video(video_path, output_dir, label=0)
            print(f"  {video_path.name}: {count} faces extracted")
        except Exception as e:
            print(f"  Error processing {video_path.name}: {e}")

# Process fake videos
fake_videos = list((data_root / "fake").glob("*.mp4")) + list((data_root / "fake").glob("*.avi"))
print(f"\nFound {len(fake_videos)} fake videos")

for video_path in tqdm(fake_videos, desc="Fake videos"):
    output_dir = output_root / "fake" / video_path.stem
    if not (output_dir / "metadata.json").exists():
        try:
            count = extract_faces_from_video(video_path, output_dir, label=1)
            print(f"  {video_path.name}: {count} faces extracted")
        except Exception as e:
            print(f"  Error processing {video_path.name}: {e}")

print("\n✓ Preprocessing complete!")
print(f"Processed data saved in: data/processed/")
