# 📸 Photo Culler

An open-source Python tool that automatically sorts photos into folders like **selected**, **blurry**, **closed_eye**, and **duplicates**.  
Perfect for photographers who want a simple, local version of tools like AfterShoot.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## ✨ Features

- **Sharpness detection** → filters out blurry images  
- **Face & eye detection** → marks images with closed eyes  
- **Duplicate detection** → finds near-identical shots  
- **Automatic organization** → outputs sorted folders  
- **Local & private** → runs on your machine only  

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/photo-culler.git
cd photo-culler
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the script
```bash
python photo_culler.py -i ./photos -o ./culled
```
### Your ./culled folder will contain:
```text
culled/
├── selected/      ✅ best photos
├── blurry/        🌫️ out of focus
├── closed_eye/    😴 eyes closed
├── duplicates/    🔄 similar shots
└── others/        ❓ uncategorized
```

## 🛠️ Command Line Options

| Option                | Short | Description                                 |
|-----------------------|-------|---------------------------------------------|
| `--input`             | `-i`  | Input folder with photos (required)         |
| `--output`            | `-o`  | Output folder for results (required)        |
| `--sharpness-threshold` |       | Sharpness sensitivity (default: 50)         |
| `--verbose`           | `-v`  | Print detailed classification info          |

## 📋 Examples of All Options

```bash
# Minimal command
python photo_culler.py -i photos -o sorted

# All options specified
python photo_culler.py \
  --input /path/to/photos \
  --output /path/to/sorted \
  --sharpness-threshold 45.5 \
  --verbose
  
```
## 📁 Output Structure

```text
output_directory/
├── selected/          # ✅ High-quality images
│   ├── IMG_001.jpg   # Sharp, open eyes, no duplicates
│   └── IMG_045.jpg
├── blurry/           # 🌫️ Low sharpness images
│   ├── IMG_012.jpg   # Below sharpness threshold
│   └── IMG_089.jpg
├── closed_eye/       # 😴 Eyes closed images
│   ├── IMG_034.jpg   # Majority of eyes closed
│   └── IMG_067.jpg
├── duplicates/       # 🔄 Near-duplicate images
│   ├── IMG_023.jpg   # Similar to images in other folders
│   └── IMG_024.jpg
└── others/           # ❓ Miscellaneous images
    ├── IMG_078.jpg   # No faces detected
    └── IMG_091.jpg   # Processing errors
```

## 🧠 Classification Logic

**Processing Flow:**
1. **Duplicate Check:** Uses perceptual hashing (pHash) with Hamming distance ≤ 8 to identify near-duplicates.
2. **Sharpness Analysis:** Multi-method approach (Laplacian, Sobel, Tenengrad) with configurable threshold.
3. **Face Detection:** Utilizes MediaPipe face mesh with a confidence threshold for accurate detection.
4. **Eye State Analysis:** Calculates Eye Aspect Ratio (EAR) to detect closed eyes in faces.

🔧 Troubleshooting

## Common Installation Issues
### Import Errors
```bash
# Error: No module named 'cv2'
pip install opencv-python

# Error: No module named 'mediapipe'  
pip install mediapipe

# Error: No module named 'PIL'
pip install pillow

# Error: No module named 'imagehash'
pip install imagehash
```
## Wrong Classifications
### Too many photos marked as blurry:
```bash
# Lower the threshold
python photo_culler.py -i photos -o sorted --sharpness-threshold 25 -v
```
### Missing closed-eye detection:
```bash
# Use verbose mode to debug
python photo_culler.py -i photos -o sorted -v
# Check eye ratio values in output
```
### Too many duplicates:

```bash
# Edit script: change hamming_distance <= 8 to <= 5
```
# Debug Mode Usage
```bash
# Run with verbose output
python photo_culler.py -i ./photos -o ./sorted -v

# Sample verbose output:
# IMG_001.jpg:
#   Category: selected
#   Reason: Passed all quality checks
#   Sharpness (combined): 67.45
#   left eye ratio: 0.234
#   right eye ratio: 0.241
#
# IMG_002.jpg:
#   Category: closed_eye
#   Reason: Majority of eyes closed: 100%
#   Sharpness (combined): 78.23
#   left eye ratio: 0.089
#   right eye ratio: 0.102
```
