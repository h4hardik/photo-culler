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


