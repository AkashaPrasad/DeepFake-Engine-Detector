"""
Multi-Face Deepfake Detection Inference
Compatible with your existing trained model from 3_train.py

This script detects ALL faces in images/videos and classifies each person separately.
Works with the model you trained on single-face Kaggle dataset.

Usage:
    python multiface_inference.py image.jpg
    python multiface_inference.py video.mp4
    
Dependencies (install if missing):
    pip install facenet-pytorch scipy
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine


# ==================== MODEL DEFINITION (Same as training) ====================
class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        if x.device.type == "mps":
            gray_cpu = gray.unsqueeze(1).cpu()
            fft = torch.fft.fft2(gray_cpu)
            magnitude = torch.abs(torch.fft.fftshift(fft)).to(x.device)
        else:
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
            'efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg'
        )
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


# ==================== MULTI-FACE DETECTION & TRACKING ====================
class MultiFaceDeepfakeDetector:
    """
    Detects multiple faces and classifies each person independently
    """
    
    def __init__(self, model_path, device=None, embedding_threshold=0.6):
        """
        Args:
            model_path: Path to your trained model (models/best_model.pth)
            device: torch device (auto-detected if None)
            embedding_threshold: Distance threshold for face identity matching (0.4-0.7)
        """
        if device is None:
            device = self.get_device()
        self.device = device
        self.embedding_threshold = embedding_threshold
        
        # Load deepfake detection model (your trained model)
        print(f"Loading deepfake model from {model_path}...")
        self.deepfake_model = DeepfakeDetector().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.deepfake_model.load_state_dict(checkpoint['model_state_dict'])
        self.deepfake_model.eval()
        
        auc = checkpoint.get('auc', 'N/A')
        print(f"✓ Model loaded (AUC: {auc})\n")
        
        # Face detector - keep_all=True to detect ALL faces
        print("Initializing MTCNN face detector...")
        detector_device = torch.device("cpu") if device.type == "mps" else device
        if detector_device.type == "cpu" and device.type == "mps":
            print("⚠️  MTCNN on MPS can fail with adaptive pooling; using CPU for face detection.")
        self.face_detector = MTCNN(
            keep_all=True,
            device=detector_device,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False  # We'll do our own preprocessing
        )
        
        # Face embedding model for tracking identities across frames
        print("Loading face recognition model for tracking...")
        self.embedding_model = None
        try:
            self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            print("✓ Face recognition model loaded")
        except Exception as exc:
            print(f"⚠️  Face recognition model not available ({exc}). Falling back to IoU tracking.")
            self.embedding_model = None
        print("✓ All models ready!\n")

    @staticmethod
    def get_device():
        """Auto-detect best available device"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    def get_face_embedding(self, face_rgb):
        """
        Generate embedding for face tracking using FaceNet
        
        Args:
            face_rgb: RGB face image (numpy array)
        Returns:
            embedding vector (512-dim)
        """
        # Resize to 160x160 for InceptionResnetV1
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
        
        # Normalize for FaceNet
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        if self.embedding_model is None:
            raise RuntimeError("Face embedding model is not available.")

        with torch.no_grad():
            embedding = self.embedding_model(face_tensor)
        
        return embedding.cpu().numpy().flatten()

    def match_identity(self, embedding, known_identities):
        """
        Match face embedding to known identities
        
        Args:
            embedding: Current face embedding
            known_identities: Dict of {identity_id: embedding}
        Returns:
            identity_id (int) - existing ID or new ID
        """
        if not known_identities:
            return 0  # First person
        
        min_dist = float('inf')
        best_match_id = None
        
        for identity_id, known_emb in known_identities.items():
            dist = cosine(embedding, known_emb)
            if dist < min_dist:
                min_dist = dist
                best_match_id = identity_id
        
        # If distance is below threshold, same person
        if min_dist < self.embedding_threshold:
            return best_match_id
        else:
            # New person detected
            return max(known_identities.keys()) + 1

    @staticmethod
    def iou_xyxy(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-8)

    @staticmethod
    def match_identity_iou(bbox, known_identities, used_ids, threshold=0.3):
        best_iou = 0.0
        best_id = None
        for identity_id, data in known_identities.items():
            if identity_id in used_ids:
                continue
            prev_bbox = data.get("bbox")
            if prev_bbox is None:
                continue
            iou = MultiFaceDeepfakeDetector.iou_xyxy(bbox, prev_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = identity_id

        if best_id is not None and best_iou >= threshold:
            return best_id
        if known_identities:
            return max(known_identities.keys()) + 1
        return 0

    def preprocess_face_for_deepfake_detection(self, face_bgr):
        """
        Preprocess face for your trained deepfake detector
        
        Args:
            face_bgr: BGR face crop from OpenCV
        Returns:
            preprocessed tensor ready for model
        """
        # Convert to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (training size)
        face_resized = cv2.resize(face_rgb, (224, 224))
        
        # Normalize (ImageNet stats - same as training)
        face_norm = face_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face_norm = (face_norm - mean) / std
        
        # Convert to tensor
        face_tensor = torch.from_numpy(face_norm).permute(2, 0, 1).unsqueeze(0).float()
        return face_tensor.to(self.device), face_rgb

    def predict_deepfake(self, face_tensor):
        """
        Predict if face is deepfake using your trained model
        
        Returns:
            fake_probability (float)
        """
        with torch.no_grad():
            outputs = self.deepfake_model(face_tensor)
            probs = torch.softmax(outputs, 1)
            fake_prob = probs[0, 1].item()
        return fake_prob

    def process_image(self, image_path):
        """
        Process single image with multiple faces
        
        Returns:
            dict with results for each detected person
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect all faces
        boxes, probs = self.face_detector.detect(img_rgb)
        
        if boxes is None or len(boxes) == 0:
            return {
                "input_type": "image",
                "input_path": str(image_path),
                "persons": [],
                "message": "No faces detected in image"
            }
        
        # Process each detected face
        persons = []
        for idx, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < 0.9:  # Skip low confidence detections
                continue
            
            # Extract face with margin
            x1, y1, x2, y2 = box.astype(int)
            h, w = img.shape[:2]
            
            margin = int((x2 - x1) * 0.3)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            
            # Predict deepfake
            face_tensor, _ = self.preprocess_face_for_deepfake_detection(face_crop)
            fake_prob = self.predict_deepfake(face_tensor)
            
            persons.append({
                "person_id": idx,
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                "is_fake": fake_prob > 0.5,
                "confidence": round(float(fake_prob), 4),
                "detection_confidence": round(float(prob), 4)
            })
        
        return {
            "input_type": "image",
            "input_path": str(image_path),
            "total_faces_detected": len(persons),
            "persons": persons
        }

    def process_video(self, video_path, sample_fps=5):
        """
        Process video with multiple faces, tracking each person
        
        Args:
            video_path: Path to video file
            sample_fps: Sample frames per second (lower = faster, higher = more accurate)
        
        Returns:
            dict with per-person results across video
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(int(round(vid_fps / sample_fps)), 1)
        
        # Track predictions per identity
        known_identities = {}  # {identity_id: {"embedding": emb, "bbox": bbox}}
        person_predictions = defaultdict(list)  # {identity_id: [fake_probs]}
        person_frame_data = defaultdict(list)  # {identity_id: [(frame_idx, timestamp, fake_prob)]}
        
        frame_idx = 0
        processed_frames = 0
        multi_face_frames = 0
        
        print(f"Processing video: {Path(video_path).name}")
        print(f"Total frames: {total_frames}, FPS: {vid_fps:.1f}, Sampling every {stride} frames\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect ALL faces in frame
                boxes, probs = self.face_detector.detect(rgb_frame)
                
                if boxes is not None and len(boxes) > 0:
                    if len(boxes) > 1:
                        multi_face_frames += 1
                    
                    used_ids = set()
                    for box, prob in zip(boxes, probs):
                        if prob < 0.9:
                            continue
                        
                        # Extract face
                        x1, y1, x2, y2 = box.astype(int)
                        h, w = frame.shape[:2]
                        
                        margin = int((x2 - x1) * 0.3)
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue
                        
                        # Match identity (embedding if available, else IoU)
                        face_tensor, face_rgb = self.preprocess_face_for_deepfake_detection(face_crop)
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        if self.embedding_model is not None:
                            embedding = self.get_face_embedding(face_rgb)
                            embeddings_only = {
                                k: v["embedding"]
                                for k, v in known_identities.items()
                                if "embedding" in v
                            }
                            identity_id = self.match_identity(embedding, embeddings_only)
                            known_identities.setdefault(identity_id, {})
                            known_identities[identity_id]["embedding"] = embedding
                        else:
                            identity_id = self.match_identity_iou(bbox, known_identities, used_ids)

                        # Update known identities
                        known_identities.setdefault(identity_id, {})
                        known_identities[identity_id]["bbox"] = bbox
                        used_ids.add(identity_id)
                        
                        # Predict deepfake
                        fake_prob = self.predict_deepfake(face_tensor)
                        person_predictions[identity_id].append(fake_prob)
                        
                        # Store frame-level data for temporal segmentation
                        timestamp = frame_idx / vid_fps
                        person_frame_data[identity_id].append((frame_idx, timestamp, fake_prob))
                
                processed_frames += 1
                
                # Progress indicator
                if processed_frames % 20 == 0:
                    print(f"Processed {processed_frames} frames, detected {len(known_identities)} unique persons...", end='\r')
            
            frame_idx += 1
        
        cap.release()
        print(f"\nCompleted processing {processed_frames} frames")
        
        # Helper function to format timestamp
        def format_timestamp(seconds):
            """Convert seconds to HH:MM:SS.mmm format"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        
        # Helper function to detect fake segments using temporal analysis
        def detect_fake_segments(frame_data, threshold=0.5, min_segment_duration=1.0, merge_gap=2.0):
            """
            Detect continuous segments where person appears as fake
            
            Args:
                frame_data: List of (frame_idx, timestamp, fake_prob)
                threshold: Probability threshold for fake detection
                min_segment_duration: Minimum duration (seconds) for a segment
                merge_gap: Merge segments if gap is less than this (seconds)
            
            Returns:
                List of segment dicts with start_time, end_time, confidence
            """
            if not frame_data:
                return []
            
            # Sort by timestamp
            frame_data = sorted(frame_data, key=lambda x: x[1])
            
            segments = []
            current_segment = None
            
            for frame_idx, timestamp, fake_prob in frame_data:
                is_fake = fake_prob > threshold
                
                if is_fake:
                    if current_segment is None:
                        # Start new segment
                        current_segment = {
                            'start_time': timestamp,
                            'end_time': timestamp,
                            'probabilities': [fake_prob]
                        }
                    else:
                        # Extend current segment
                        current_segment['end_time'] = timestamp
                        current_segment['probabilities'].append(fake_prob)
                else:
                    if current_segment is not None:
                        # End current segment
                        segments.append(current_segment)
                        current_segment = None
            
            # Don't forget last segment
            if current_segment is not None:
                segments.append(current_segment)
            
            # Filter out segments shorter than minimum duration
            valid_segments = []
            for seg in segments:
                duration = seg['end_time'] - seg['start_time']
                if duration >= min_segment_duration:
                    valid_segments.append(seg)
            
            # Merge nearby segments
            if len(valid_segments) <= 1:
                merged_segments = valid_segments
            else:
                merged_segments = [valid_segments[0]]
                for seg in valid_segments[1:]:
                    prev_seg = merged_segments[-1]
                    gap = seg['start_time'] - prev_seg['end_time']
                    
                    if gap <= merge_gap:
                        # Merge with previous segment
                        prev_seg['end_time'] = seg['end_time']
                        prev_seg['probabilities'].extend(seg['probabilities'])
                    else:
                        merged_segments.append(seg)
            
            # Calculate average confidence for each segment
            result_segments = []
            for seg in merged_segments:
                avg_confidence = float(np.mean(seg['probabilities']))
                result_segments.append({
                    'start_time': format_timestamp(seg['start_time']),
                    'end_time': format_timestamp(seg['end_time']),
                    'confidence': round(avg_confidence, 4)
                })
            
            return result_segments
        
        # Aggregate results per person
        persons = []
        for identity_id in sorted(person_predictions.keys()):
            probs = person_predictions[identity_id]
            frame_data = person_frame_data[identity_id]
            
            avg_prob = np.mean(probs)
            max_prob = np.max(probs)
            min_prob = np.min(probs)
            is_fake = bool(avg_prob > 0.5)
            
            # Detect fake segments for this person
            fake_segments = []
            if is_fake:  # Only generate segments for fake persons
                fake_segments = detect_fake_segments(
                    frame_data,
                    threshold=0.5,
                    min_segment_duration=1.0,
                    merge_gap=2.0
                )
            
            persons.append({
                "person_id": identity_id,
                "is_fake": is_fake,
                "confidence": round(float(avg_prob), 4),
                "confidence_range": {
                    "min": round(float(min_prob), 4),
                    "max": round(float(max_prob), 4)
                },
                "num_detections": len(probs),
                "fake_segments": fake_segments  # Add temporal segments
            })
        
        return {
            "input_type": "video",
            "input_path": str(video_path),
            "video_info": {
                "total_frames": total_frames,
                "fps": round(vid_fps, 2),
                "processed_frames": processed_frames
            },
            "summary": {
                "total_persons_detected": len(persons),
                "fake_persons": sum(1 for p in persons if p["is_fake"]),
                "real_persons": sum(1 for p in persons if not p["is_fake"]),
                "frames_with_multiple_faces": multi_face_frames
            },
            "persons": persons
        }


# ==================== MAIN EXECUTION ====================
def main():
    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════════╗
║         Multi-Face Deepfake Detection System                  ║
╚════════════════════════════════════════════════════════════════╝

This system detects ALL faces in images/videos and classifies
each person independently as REAL or FAKE.

Usage:
    python multiface_inference.py <path_to_image_or_video>

Examples:
    python multiface_inference.py test_image.jpg
    python multiface_inference.py interview_video.mp4

Features:
    ✓ Detects multiple faces per frame
    ✓ Tracks each person across video
    ✓ Independent classification per person
    ✓ Works with your trained model

Required files:
    - models/best_model.pth (your trained model)

Dependencies (install if missing):
    pip install facenet-pytorch scipy
        """)
        sys.exit(0)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)
    
    # Check if model exists
    model_path = Path("models/best_model_celeb.pth")
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train your model first using 3_train.py")
        sys.exit(1)
    
    # Initialize detector
    print("=" * 70)
    print("MULTI-FACE DEEPFAKE DETECTION")
    print("=" * 70 + "\n")
    
    detector = MultiFaceDeepfakeDetector(model_path)
    
    # Detect input type
    suffix = input_path.suffix.lower()
    
    if suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process image
        result = detector.process_image(input_path)
    elif suffix in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process video
        result = detector.process_video(input_path, sample_fps=5)
    else:
        print(f"❌ Unsupported file format: {suffix}")
        print("Supported: Images (.jpg, .png) or Videos (.mp4, .avi, .mov)")
        sys.exit(1)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70 + "\n")
    
    def fmt_conf(val):
        return f"{float(val):.4f}"

    if result["input_type"] == "image":
        print(f"📷 Image: {result['input_path']}")
        print(f"👥 Faces detected: {result['total_faces_detected']}\n")
        
        if result['total_faces_detected'] == 0:
            print(result.get('message', 'No faces detected'))
        else:
            for person in result['persons']:
                status = "🔴 FAKE" if person['is_fake'] else "🟢 REAL"
                print(f"Person {person['person_id']}: {status}")
                print(f"  Confidence: {fmt_conf(person['confidence'])}")
                print(f"  Bounding box: {person['bounding_box']}\n")
    
    else:  # video
        print(f"🎬 Video: {result['input_path']}")
        print(f"📊 Total frames: {result['video_info']['total_frames']}")
        print(f"⚡ FPS: {result['video_info']['fps']}")
        print(f"✓ Processed: {result['video_info']['processed_frames']} frames\n")
        
        print("📈 Summary:")
        print(f"  Total persons: {result['summary']['total_persons_detected']}")
        print(f"  🔴 Fake: {result['summary']['fake_persons']}")
        print(f"  🟢 Real: {result['summary']['real_persons']}")
        print(f"  Multi-face frames: {result['summary']['frames_with_multiple_faces']}\n")
        
        print("👥 Per-Person Results:")
        for person in result['persons']:
            status = "🔴 FAKE" if person['is_fake'] else "🟢 REAL"
            print(f"\n  Person {person['person_id']}: {status}")
            print(f"    Average confidence: {fmt_conf(person['confidence'])}")
            print(f"    Confidence range: {fmt_conf(person['confidence_range']['min'])} - {fmt_conf(person['confidence_range']['max'])}")
            print(f"    Detected in {person['num_detections']} frames")
            
            # Display fake segments if person is fake
            if person['is_fake'] and person.get('fake_segments'):
                print(f"\n    🎬 Fake Segments (timestamp ranges):")
                for idx, segment in enumerate(person['fake_segments'], 1):
                    print(f"      Segment {idx}: {segment['start_time']} → {segment['end_time']}")
                    print(f"                  Confidence: {fmt_conf(segment['confidence'])}")
    
    # Save results to JSON
    output_file = input_path.stem + "_multiface_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
