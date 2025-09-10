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
### ⚙️ Advanced Configuration
## Custom Thresholds in Code
#### Edit these values in the script for fine-tuning:
```python
# Eye detection sensitivity (line ~150)
if left_ear < 0.15:  # Lower = more sensitive to closed eyes

# Closed eye majority threshold (line ~280)
if closed_ratio > 0.6:  # Lower = stricter about closed eyes

# Duplicate detection sensitivity (line ~320)
if hamming_distance <= 8:  # Lower = stricter duplicate detection

# Sharpness calculation weights (line ~95)
combined_score = (laplacian_var * 0.6 + sobel_mean * 0.4)
```
### Adding Custom Categories
#### 1.Modify the categories list:
```python
categories = ['selected', 'blurry', 'closed_eye', 'duplicates', 'others', 'custom_category']
```
#### 2.Implement classification logic:
```python
def classify_image(self, image_path, duplicates):
    # ... existing logic ...
    
    # Your custom logic
    if your_custom_condition:
        return 'custom_category', debug_info
```
### Integration Examples
#### Shell Script Wrapper
```bash
#!/bin/bash
#!/bin/bash
# auto_cull.sh - Wrapper script for photo culling

PHOTOS_DIR="\$1"
OUTPUT_DIR="\$2"
THRESHOLD="${3:-50}"

if [ -z "$$PHOTOS_DIR" ] || [ -z "$$OUTPUT_DIR" ]; then
    echo "Usage: \$0 <photos_directory> <output_directory> [threshold]"
    echo "Example: \$0 ./photos ./sorted 45"
    exit 1
fi

echo "🚀 Starting photo culling..."
echo "📁 Input: $PHOTOS_DIR"
echo "📁 Output: $OUTPUT_DIR"
echo "🎯 Threshold: $THRESHOLD"

python photo_culler.py \
    -i "$PHOTOS_DIR" \
    -o "$OUTPUT_DIR" \
    --sharpness-threshold "$THRESHOLD" \
    -v

if [ $? -eq 0 ]; then
    echo "✅ Culling completed successfully!"
    echo "📊 Results:"
    ls -la "$OUTPUT_DIR"
else
    echo "❌ Culling failed!"
    exit 1
fi
```
#### Python Integration
```python
import subprocess
import sys
from pathlib import Path

def cull_photos(input_dir, output_dir, threshold=50.0, verbose=False):
    """
    Wrapper function to call photo culler from another Python script.
    
    Returns:
        tuple: (success, stdout, stderr)
    """
    cmd = [
        sys.executable, 'photo_culler.py',
        '-i', str(input_dir),
        '-o', str(output_dir),
        '--sharpness-threshold', str(threshold)
    ]
    
    if verbose:
        cmd.append('-v')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Process timed out after 1 hour"

# Usage example
if __name__ == "__main__":
    success, output, error = cull_photos("./photos", "./sorted", threshold=45, verbose=True)
    
    if success:
        print("✅ Photo culling completed!")
    else:
        print(f"❌ Error: {error}")
```
# 🚄 Performance Tips
## Speed Optimization
#### 1.Reduce MediaPipe accuracy for speed:
```python
min_detection_confidence=0.1  # Default: 0.3
```
#### 2.Process images in batches:
```python
# Instead of processing 10,000 images at once
find photos -name "*.jpg" | split -l 1000 - batch_
for batch in batch_*; do
    # Process each batch separately
done

```    
## 🧠 Memory Management

- **Large collections:** Process images in batches of 1000–2000 for efficiency.
- **High-resolution images:** Resize images for analysis to reduce memory usage.
- **Multiple runs:** Clear output directories before each run to avoid mixing results.

## ⚖️ Quality vs Speed Trade-offs

| Setting                      | Speed   | Accuracy | Recommended Use Case   |
|------------------------------|---------|----------|-----------------------|
| `min_detection_confidence=0.1` | Fast    | Lower    | Quick sorting         |
| `min_detection_confidence=0.3` | Medium  | Good     | Balanced              |
| `min_detection_confidence=0.5` | Slow    | High     | Precise sorting       |

## 📊 Example Output
### Console Output
```commandline
🚀 Starting improved photo culling process...
📁 Input directory: ./photos
📁 Output directory: ./culled
🎯 Sharpness threshold: 50.0

Setting up output directories in: ./culled
  ✓ Created: selected/
  ✓ Created: blurry/
  ✓ Created: closed_eye/
  ✓ Created: duplicates/
  ✓ Created: others/

Scanning for images in: ./photos
Found 150 image files

Calculating perceptual hashes for duplicate detection...
Found 12 duplicate images

Processing 150 images...
Processed 50/150 images
Processed 100/150 images
Processed 150/150 images

✅ Photo culling completed!
```
## JSON Summary Output
```commandline
{
  "improved_photo_culling_summary": {
    "input_directory": "./photos",
    "output_directory": "./culled",
    "sharpness_threshold": 50.0,
    "statistics": {
      "total_processed": 150,
      "selected": 89,
      "blurry": 23,
      "closed_eye": 15,
      "duplicates": 18,
      "others": 5
    },
    "percentages": {
      "selected": 59.3,
      "blurry": 15.3,
      "closed_eye": 10.0,
      "duplicates": 12.0,
      "others": 3.3
    }
  }
}
```

## 🤝 Contributing

### Bug Reports
When reporting bugs, please include:
- **Python version:** `python --version`
- **Operating system:** Windows/macOS/Linux + version
- **Package versions:** `pip list | grep -E "(opencv|mediapipe|pillow|imagehash)"`
- **Full error message with stack trace**
- **Command used and expected vs actual behavior**

### Feature Requests
- **Describe the use case and problem it solves**
- **Provide example scenarios**
- **Suggest implementation approach if you have ideas**

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **OpenCV** for computer vision algorithms
- **MediaPipe** for face and landmark detection
- **ImageHash** for perceptual hashing
- **Pillow** for image processing

# 📸 Photo Culler

**Last updated:** 2024  
**Version:** 2.1

An open-source Python tool that automatically sorts photos into folders like **selected**, **blurry**, **closed_eye**, and **duplicates**.
Perfect for photographers who want a simple, local version of tools.

...

# RAW Version (Professional)
```commandline
# Process RAW files keeping original format
python photo_culler_raw.py -i ./raw_photos -o ./sorted_raw -v

# Convert RAW to JPEG while sorting
python photo_culler_raw.py -i ./raw_photos -o ./sorted_jpeg --convert-to-jpeg --jpeg-quality 95

# Professional workflow with high standards
python photo_culler_raw.py -i ./shoot_photos -o ./processed --sharpness-threshold 50 --convert-to-jpeg -v
```

🎉 **Happy Photo Culling!**  

Transform your chaotic photo collection into an organized masterpiece! 📸✨
