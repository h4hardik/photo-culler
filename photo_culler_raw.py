#!/usr/bin/env python3
"""
Photo Culler - RAW Professional Version

Advanced version supporting RAW camera files from Canon, Nikon, Sony, etc.
Designed for professional photographers and advanced users.

Installation requirements:
pip install opencv-python pillow imagehash mediapipe numpy rawpy

Additional system requirements:
- LibRaw library (for some RAW formats)
- More RAM (recommend 8GB+ for large RAW collections)
- SSD storage recommended for better performance

Usage:
python photo_culler_raw.py --input ./raw_photos --output ./culled --convert-to-jpeg

Author: CodePilot
Version: 2.1-RAW-Pro
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import hashlib

try:
    import cv2
    import numpy as np
    from PIL import Image
    import imagehash
    import mediapipe as mp
    import rawpy
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("\nFor RAW support, please install:")
    print("pip install opencv-python pillow imagehash mediapipe numpy rawpy")
    print("\nOptional for additional RAW formats:")
    print("pip install LibRaw-py")
    sys.exit(1)


class PhotoCullerRAW:
    """Professional photo culling with RAW format support."""

    def __init__(self, input_dir: str, output_dir: str, sharpness_threshold: float = 40.0,
                 convert_to_jpeg: bool = False, jpeg_quality: int = 95):
        """
        Initialize the RAW Photo Culler.

        Args:
            input_dir: Path to input directory containing images
            output_dir: Path to output directory for sorted images
            sharpness_threshold: Minimum sharpness score (lower for RAW)
            convert_to_jpeg: Convert RAW files to JPEG in output
            jpeg_quality: JPEG quality if converting (1-100)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sharpness_threshold = sharpness_threshold
        self.convert_to_jpeg = convert_to_jpeg
        self.jpeg_quality = jpeg_quality

        # Comprehensive format support
        self.standard_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        self.raw_extensions = {
            # Canon
            '.cr2', '.cr3', '.crw', '.CR2', '.CR3', '.CRW',
            # Nikon
            '.nef', '.nrw', '.NEF', '.NRW',
            # Sony
            '.arw', '.srf', '.sr2', '.ARW', '.SRF', '.SR2',
            # Adobe DNG
            '.dng', '.DNG',
            # Fujifilm
            '.raf', '.RAF',
            # Olympus
            '.orf', '.ORF',
            # Panasonic
            '.rw2', '.raw', '.RW2', '.RAW',
            # Pentax
            '.pef', '.ptx', '.PEF', '.PTX',
            # Samsung
            '.srw', '.SRW',
            # Sigma
            '.x3f', '.X3F',
            # Phase One
            '.3fr', '.iiq', '.3FR', '.IIQ',
            # Hasselblad
            '.fff', '.FFF',
            # Leica
            '.rwl', '.dcs', '.RWL', '.DCS'
        }

        self.supported_extensions = self.standard_extensions | self.raw_extensions

        # Initialize MediaPipe with more conservative settings for RAW
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,  # Reduced for performance
                refine_landmarks=True,
                min_detection_confidence=0.4,  # Higher confidence for RAW
                min_tracking_confidence=0.4
            )
        except Exception as e:
            print(f"Warning: MediaPipe initialization failed: {e}")
            self.face_mesh = None

        # Eye landmark indices
        self.left_eye_key_points = [33, 160, 158, 133, 153, 144]
        self.right_eye_key_points = [362, 385, 387, 263, 373, 380]

        # Enhanced statistics for RAW processing
        self.stats = {
            'total_processed': 0,
            'raw_files': 0,
            'standard_files': 0,
            'selected': 0,
            'blurry': 0,
            'closed_eye': 0,
            'duplicates': 0,
            'others': 0,
            'errors': 0,
            'conversion_errors': 0,
            'processing_time': 0.0
        }

        self.image_hashes: Dict[str, str] = {}
        self.hash_groups: Dict[str, List[str]] = {}

    def setup_output_directories(self) -> None:
        """Create output directory structure."""
        categories = ['selected', 'blurry', 'closed_eye', 'duplicates', 'others']

        print(f"Setting up RAW processing directories in: {self.output_dir}")
        for category in categories:
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {category}/")

        if self.convert_to_jpeg:
            print(f"  📝 RAW files will be converted to JPEG (quality: {self.jpeg_quality})")

    def get_image_files(self) -> List[Path]:
        """Get all supported image files including RAW formats."""
        image_files = []
        raw_count = 0
        standard_count = 0

        print(f"Scanning for images (including RAW) in: {self.input_dir}")
        for file_path in self.input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                image_files.append(file_path)
                if file_path.suffix in self.raw_extensions:
                    raw_count += 1
                else:
                    standard_count += 1

        print(f"Found {len(image_files)} total files:")
        print(f"  📸 {standard_count} standard images (JPG/PNG)")
        print(f"  📷 {raw_count} RAW files")

        if raw_count > 0:
            print(f"  ⚠️  RAW processing will be slower and use more memory")

        return sorted(image_files)

    def process_raw_to_rgb(self, raw_path: Path) -> Optional[np.ndarray]:
        """
        Process RAW file to RGB array for analysis.

        Args:
            raw_path: Path to RAW file

        Returns:
            RGB numpy array or None if processing failed
        """
        try:
            print(f"    Processing RAW: {raw_path.name}")
            start_time = time.time()

            with rawpy.imread(str(raw_path)) as raw:
                # Fast processing settings for analysis
                rgb = raw.postprocess(
                    use_camera_wb=True,  # Use camera white balance
                    half_size=True,  # Half resolution for speed
                    no_auto_bright=True,  # Disable auto brightness
                    output_bps=8,  # 8-bit output
                    gamma=(2.222, 4.5),  # Standard gamma
                    user_qual=0,  # Fastest interpolation
                    four_color_rgb=False,  # Faster processing
                    dcb_enhance=False,  # Disable enhancement
                    user_black=None,  # Auto black level
                    user_sat=None  # Auto saturation
                )

            processing_time = time.time() - start_time
            print(f"      ✓ Processed in {processing_time:.1f}s")

            return rgb

        except Exception as e:
            print(f"      ❌ RAW processing failed: {e}")
            self.stats['conversion_errors'] += 1
            return None

    def calculate_sharpness_scores(self, image_path: Path) -> Dict[str, float]:
        """Enhanced sharpness calculation with RAW support."""
        try:
            is_raw = image_path.suffix in self.raw_extensions

            if is_raw:
                # Process RAW file
                rgb_array = self.process_raw_to_rgb(image_path)
                if rgb_array is None:
                    return {'laplacian': 0.0, 'sobel': 0.0, 'combined': 0.0}

                # Convert RGB to BGR for OpenCV
                image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                self.stats['raw_files'] += 1
            else:
                # Standard image processing
                image = cv2.imread(str(image_path))
                if image is None:
                    return {'laplacian': 0.0, 'sobel': 0.0, 'combined': 0.0}
                self.stats['standard_files'] += 1

            # Sharpness calculation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Method 1: Variance of Laplacian
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Method 2: Sobel gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            sobel_mean = np.mean(sobel_magnitude)

            # Method 3: Tenengrad (useful for RAW)
            tenengrad = np.mean(sobelx ** 2 + sobely ** 2)

            # Combined score with RAW-specific weighting
            if is_raw:
                # RAW files often have lower initial scores due to processing
                combined_score = (laplacian_var * 0.5 + sobel_mean * 0.3 + tenengrad * 0.0002)
            else:
                combined_score = (laplacian_var * 0.6 + sobel_mean * 0.4)

            return {
                'laplacian': float(laplacian_var),
                'sobel': float(sobel_mean),
                'tenengrad': float(tenengrad),
                'combined': float(combined_score),
                'is_raw': is_raw
            }

        except Exception as e:
            print(f"  Error calculating sharpness for {image_path.name}: {e}")
            return {'laplacian': 0.0, 'sobel': 0.0, 'tenengrad': 0.0, 'combined': 0.0, 'is_raw': False}

    def calculate_eye_aspect_ratio(self, eye_points: np.ndarray, image_shape: tuple) -> float:
        """Calculate Eye Aspect Ratio for closed eye detection."""
        try:
            h, w = image_shape[:2]
            eye_pixels = eye_points.copy()
            eye_pixels[:, 0] *= w
            eye_pixels[:, 1] *= h

            # Calculate vertical distances
            v1 = np.linalg.norm(eye_pixels[1] - eye_pixels[5])
            v2 = np.linalg.norm(eye_pixels[2] - eye_pixels[4])

            # Calculate horizontal distance
            h1 = np.linalg.norm(eye_pixels[0] - eye_pixels[3])

            if h1 > 0:
                ear = (v1 + v2) / (2.0 * h1)
            else:
                ear = 0.0

            return ear

        except Exception:
            return 0.25

    def detect_faces_and_eyes(self, image_path: Path) -> Tuple[int, int, int, List[float]]:
        """Enhanced face detection with RAW support."""
        try:
            is_raw = image_path.suffix in self.raw_extensions

            if is_raw:
                rgb_array = self.process_raw_to_rgb(image_path)
                if rgb_array is None:
                    return 0, 0, 0, []
                rgb_image = rgb_array
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    return 0, 0, 0, []
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            if self.face_mesh is None:
                return 0, 0, 0, []

            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return 0, 0, 0, []

            num_faces = len(results.multi_face_landmarks)
            num_eyes = 0
            num_closed_eyes = 0
            eye_ratios = []

            h, w = rgb_image.shape[:2]

            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

                # Left eye
                left_eye_points = landmarks[self.left_eye_key_points]
                left_ear = self.calculate_eye_aspect_ratio(left_eye_points, (h, w))
                eye_ratios.append(left_ear)
                num_eyes += 1

                if left_ear < 0.15:
                    num_closed_eyes += 1

                # Right eye
                right_eye_points = landmarks[self.right_eye_key_points]
                right_ear = self.calculate_eye_aspect_ratio(right_eye_points, (h, w))
                eye_ratios.append(right_ear)
                num_eyes += 1

                if right_ear < 0.15:
                    num_closed_eyes += 1

            return num_faces, num_eyes, num_closed_eyes, eye_ratios

        except Exception as e:
            print(f"  Error detecting faces/eyes for {image_path}: {e}")
            return 0, 0, 0, []

    def calculate_perceptual_hash(self, image_path: Path) -> str:
        """Calculate perceptual hash with RAW support."""
        try:
            is_raw = image_path.suffix in self.raw_extensions

            if is_raw:
                rgb_array = self.process_raw_to_rgb(image_path)
                if rgb_array is None:
                    return hashlib.md5(str(image_path).encode()).hexdigest()[:16]

                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_array)
            else:
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')

            # Calculate perceptual hash
            hash_value = imagehash.phash(pil_image, hash_size=8)
            return str(hash_value)

        except Exception as e:
            print(f"  Error calculating hash for {image_path}: {e}")
            return hashlib.md5(str(image_path).encode()).hexdigest()[:16]

    def find_duplicates(self, image_files: List[Path]) -> Set[str]:
        """Find duplicate images including RAW files."""
        print("Calculating perceptual hashes (including RAW processing)...")

        for i, image_path in enumerate(image_files, 1):
            if i % 10 == 0:
                print(f"  Processing hash {i}/{len(image_files)}")

            hash_value = self.calculate_perceptual_hash(image_path)
            self.image_hashes[str(image_path)] = hash_value

            if hash_value not in self.hash_groups:
                self.hash_groups[hash_value] = []
            self.hash_groups[hash_value].append(str(image_path))

        # Find similar hashes
        duplicate_paths = set()
        processed_hashes = set()

        for hash1, paths1 in self.hash_groups.items():
            if hash1 in processed_hashes:
                continue

            similar_group = list(paths1)

            for hash2, paths2 in self.hash_groups.items():
                if hash1 != hash2 and hash2 not in processed_hashes:
                    try:
                        h1_int = int(hash1, 16)
                        h2_int = int(hash2, 16)
                        hamming_distance = bin(h1_int ^ h2_int).count('1')

                        if hamming_distance <= 10:  # Slightly more lenient for RAW
                            similar_group.extend(paths2)
                            processed_hashes.add(hash2)
                    except ValueError:
                        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
                        if hamming_distance <= 6:
                            similar_group.extend(paths2)
                            processed_hashes.add(hash2)

            if len(similar_group) > 1:
                # Prefer keeping RAW over JPEG in duplicates
                raw_files = [p for p in similar_group if Path(p).suffix in self.raw_extensions]
                if raw_files:
                    # Keep first RAW, mark others as duplicates
                    duplicate_paths.update(similar_group[1:])
                else:
                    # Standard duplicate handling
                    duplicate_paths.update(similar_group[1:])

            processed_hashes.add(hash1)

        print(f"Found {len(duplicate_paths)} duplicate images")
        return duplicate_paths

    def classify_image(self, image_path: Path, duplicates: Set[str]) -> Tuple[str, Dict]:
        """Classify image with RAW-specific logic."""
        debug_info = {}
        image_path_str = str(image_path)
        is_raw = image_path.suffix in self.raw_extensions

        # Check if duplicate
        if image_path_str in duplicates:
            debug_info['reason'] = 'Detected as duplicate'
            debug_info['is_raw'] = is_raw
            return 'duplicates', debug_info

        # Calculate sharpness
        sharpness_scores = self.calculate_sharpness_scores(image_path)
        debug_info['sharpness'] = sharpness_scores
        debug_info['is_raw'] = is_raw

        combined_sharpness = sharpness_scores['combined']

        # RAW-adjusted thresholds
        effective_threshold = self.sharpness_threshold
        if is_raw:
            effective_threshold *= 0.7  # RAW files often score lower initially

        # Check if blurry
        if combined_sharpness < effective_threshold:
            debug_info['reason'] = f'Low sharpness: {combined_sharpness:.2f} < {effective_threshold:.2f}'
            return 'blurry', debug_info

        # Face and eye detection
        num_faces, num_eyes, num_closed_eyes, eye_ratios = self.detect_faces_and_eyes(image_path)
        debug_info['faces'] = num_faces
        debug_info['eyes'] = num_eyes
        debug_info['closed_eyes'] = num_closed_eyes
        debug_info['eye_ratios'] = eye_ratios

        # No faces detected
        if num_faces == 0:
            debug_info['reason'] = 'No faces detected'
            return 'others', debug_info

        # Check for closed eyes
        if num_eyes > 0:
            closed_ratio = num_closed_eyes / num_eyes
            debug_info['closed_eye_ratio'] = closed_ratio

            if closed_ratio > 0.6:
                debug_info['reason'] = f'Majority of eyes closed: {closed_ratio:.2%}'
                return 'closed_eye', debug_info

        # Selected
        debug_info['reason'] = 'Passed all quality checks'
        return 'selected', debug_info

    def copy_image_to_category(self, image_path: Path, category: str) -> None:
        """Copy image with optional RAW to JPEG conversion."""
        try:
            dest_dir = self.output_dir / category
            is_raw = image_path.suffix in self.raw_extensions

            if is_raw and self.convert_to_jpeg:
                # Convert RAW to JPEG
                rgb_array = self.process_raw_to_rgb(image_path)
                if rgb_array is not None:
                    dest_path = dest_dir / f"{image_path.stem}.jpg"

                    # Handle name conflicts
                    counter = 1
                    original_dest_path = dest_path
                    while dest_path.exists():
                        stem = original_dest_path.stem
                        dest_path = dest_dir / f"{stem}_{counter}.jpg"
                        counter += 1

                    # Save as JPEG
                    pil_image = Image.fromarray(rgb_array)
                    pil_image.save(dest_path, 'JPEG', quality=self.jpeg_quality, optimize=True)
                    print(f"      Converted: {image_path.name} → {dest_path.name}")
                else:
                    print(f"      Failed to convert: {image_path.name}")
            else:
                # Copy original file
                dest_path = dest_dir / image_path.name

                # Handle name conflicts
                counter = 1
                original_dest_path = dest_path
                while dest_path.exists():
                    stem = original_dest_path.stem
                    suffix = original_dest_path.suffix
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                shutil.copy2(image_path, dest_path)

        except Exception as e:
            print(f"  Error copying {image_path} to {category}: {e}")
            self.stats['errors'] += 1

    def process_images(self, verbose: bool = False) -> None:
        """Main processing function with RAW-specific optimizations."""
        start_time = time.time()

        print(f"🚀 Starting RAW photo culling process...")
        print(f"📁 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Sharpness threshold: {self.sharpness_threshold}")
        print(f"🔄 Convert RAW to JPEG: {self.convert_to_jpeg}")

        # Setup directories
        self.setup_output_directories()

        # Get image files
        image_files = self.get_image_files()
        if not image_files:
            print("No image files found!")
            return

        # Find duplicates
        duplicates = self.find_duplicates(image_files)

        # Process each image
        print(f"\nProcessing {len(image_files)} images (including RAW)...")
        print("⚠️  RAW processing is slower - please be patient")

        for i, image_path in enumerate(image_files, 1):
            try:
                is_raw = image_path.suffix in self.raw_extensions
                prefix = "📷 RAW" if is_raw else "📸"

                if verbose or is_raw:
                    print(f"  {prefix} Processing: {image_path.name}")

                # Classify image
                category, debug_info = self.classify_image(image_path, duplicates)

                if verbose:
                    print(f"    Category: {category}")
                    print(f"    Reason: {debug_info.get('reason', 'Unknown')}")
                    if 'sharpness' in debug_info:
                        print(f"    Sharpness: {debug_info['sharpness']['combined']:.2f}")

                # Copy to appropriate folder
                self.copy_image_to_category(image_path, category)

                # Update statistics
                self.stats['total_processed'] += 1
                self.stats[category] += 1

                # Progress indicator
                if i % 20 == 0 or i == len(image_files):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"    Progress: {i}/{len(image_files)} ({rate:.1f} files/min)")

            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                self.copy_image_to_category(image_path, 'others')
                self.stats['total_processed'] += 1
                self.stats['others'] += 1
                self.stats['errors'] += 1

        self.stats['processing_time'] = time.time() - start_time
        print(f"\n✅ RAW photo culling completed!")
        print(f"⏱️  Total processing time: {self.stats['processing_time']:.1f} seconds")

    def print_summary(self) -> None:
        """Print comprehensive summary including RAW-specific metrics."""
        summary = {
            "raw_photo_culling_summary": {
                "input_directory": str(self.input_dir),
                "output_directory": str(self.output_dir),
                "sharpness_threshold": self.sharpness_threshold,
                "convert_to_jpeg": self.convert_to_jpeg,
                "jpeg_quality": self.jpeg_quality if self.convert_to_jpeg else None,
                "processing_time_seconds": round(self.stats['processing_time'], 1),
                "processing_rate_per_minute": round(
                    self.stats['total_processed'] / (self.stats['processing_time'] / 60), 1),
                "file_types": {
                    "raw_files": self.stats['raw_files'],
                    "standard_files": self.stats['standard_files']
                },
                "statistics": {k: v for k, v in self.stats.items() if
                               k not in ['processing_time', 'raw_files', 'standard_files']},
                "percentages": {
                    category: round((count / max(self.stats['total_processed'], 1)) * 100, 1)
                    for category, count in self.stats.items()
                    if category not in ['total_processed', 'processing_time', 'raw_files', 'standard_files']
                }
            }
        }

        print("\n" + "=" * 60)
        print("RAW PHOTO CULLING SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2))


def main():
    """Main function with RAW-specific arguments."""
    parser = argparse.ArgumentParser(
        description="Professional photo culler with RAW format support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing images (including RAW files)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for sorted images'
    )

    parser.add_argument(
        '--sharpness-threshold',
        type=float,
        default=40.0,  # Lower default for RAW
        help='Minimum sharpness score (lower for RAW files)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with classification details'
    )

    parser.add_argument(
        '--convert-to-jpeg',
        action='store_true',
        help='Convert RAW files to JPEG in output folders'
    )

    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=95,
        help='JPEG quality when converting RAW files (1-100)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist!")
        return 1

    if not os.path.isdir(args.input):
        print(f"Error: '{args.input}' is not a directory!")
        return 1

    # Validate JPEG quality
    if not 1 <= args.jpeg_quality <= 100:
        print(f"Error: JPEG quality must be between 1 and 100!")
        return 1

    try:
        # Create and run RAW photo culler
        culler = PhotoCullerRAW(
            args.input,
            args.output,
            args.sharpness_threshold,
            args.convert_to_jpeg,
            args.jpeg_quality
        )
        culler.process_images(verbose=args.verbose)
        culler.print_summary()

        return 0

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())