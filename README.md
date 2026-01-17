# DeepFake Engine v2

DeepFake Engine v2 is a minimal end-to-end deepfake detection pipeline that allows you to train your own model and run inference on images (and extend to videos). It is designed to work on macOS (Apple Silicon / MPS), CUDA-enabled GPUs, or CPU, and is suitable for small datasets and rapid experimentation

---

## Architecture Overview

The system follows a simple but effective forensic pipeline:

1. **Video Input**
   - Real and fake videos are placed manually into separate folders.

2. **Frame Extraction**
   - Videos are sampled at a fixed FPS to extract frames.

3. **Face Preprocessing**
   - Faces are detected and cropped from frames.
   - Cropped faces are resized and normalized.

4. **Deepfake Detection Model**
   - Dual-branch neural network:
     - RGB branch: EfficientNet-B0 backbone for spatial artifacts.
     - Frequency branch: FFT-based features to capture GAN artifacts.
   - Features are fused and passed to a classifier.
   - Output is a probability of fake vs real.

5. **Inference**
   - Single-image inference returns fake/real with confidence.
   - Video inference can be extended by aggregating frame-level predictions.

---

## Requirements

- Python 3.9 or higher
- One of the following:
  - NVIDIA GPU with CUDA
  - Apple Silicon (MPS)
  - CPU

---

## Setup

```bash
cd /Users/akashaaprasad/Documents/DeepFake\ Engine_v2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```bat
cd C:\path\to\DeepFake Engine_v2
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install a CUDA-enabled PyTorch build that matches your driver/toolkit:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Data Download

This repo does not automatically download datasets. Put your videos here:

```
data/
  raw/
    real/   # real videos (.mp4/.avi)
    fake/   # deepfake videos (.mp4/.avi)
```

You can run the helper instructions:

```bash
python download_data.py
```
## Extract Frames

```bash
python extract_frames.py
```



## Preprocess (Extract Faces)

```bash
python preprocess.py
```

This creates:

```
data/
  processed/
    real/<video_name>/*.jpg
    fake/<video_name>/*.jpg
```

## Train

```bash
python train.py
```

The best checkpoint is saved to:

```
models/best_model.pth
```

The training script uses CUDA if available; otherwise it falls back to MPS or CPU.

## Inference (Single Image)

```bash
python inference.py /full/path/to/image.jpg
```

Example:

```bash
python inference.py /Users/akashaaprasad/Downloads/test4.jpeg
```

If you see an error like `Could not read image`, the file path is wrong or the image cannot be opened.

## Notes

- If you retrain the model, inference will use the new `models/best_model.pth`.
- For faster iteration, use a small set of videos.
