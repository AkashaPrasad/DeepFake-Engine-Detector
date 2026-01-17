"""
Download a small subset of data for testing
"""
import os
import urllib.request
from pathlib import Path

# Create data directories
Path("data/raw/real").mkdir(parents=True, exist_ok=True)
Path("data/raw/fake").mkdir(parents=True, exist_ok=True)

print("""
=================================================================
DATASET DOWNLOAD INSTRUCTIONS
=================================================================

For testing, you have two options:

OPTION 1: Use Sample Videos (Quickest - 5 minutes)
--------------------------------------------------
1. Download 10-20 real videos from YouTube (any face videos)
2. Download sample deepfakes from: 
   https://github.com/ondyari/FaceForensics (sample videos)
3. Put them in:
   - data/raw/real/     (your real videos)
   - data/raw/fake/     (deepfake videos)

OPTION 2: Download Full Dataset (Better results - 1-2 hours)
-------------------------------------------------------------
FaceForensics++ Dataset:

1. Go to: https://github.com/ondyari/FaceForensics
2. Fill the Google Form to request download access
3. You'll receive an email with download script in 1-3 days
4. Run their download script:
   python download-FaceForensics.py data/raw -d Deepfakes -c c23 -t videos

This downloads ~200 videos (real + fake), about 15GB.

=================================================================

After downloading, your structure should look like:

data/
  raw/
    real/
      video1.mp4
      video2.mp4
      ...
    fake/
      video1.mp4
      video2.mp4
      ...

Press Enter when you've downloaded the videos...
""")

input()
print("✓ Ready to preprocess!")