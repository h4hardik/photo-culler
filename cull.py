#!/usr/bin/env python3
"""
Improved Photo Culling Script

Fixed issues:
- Better sharpness calculation with multiple methods
- Improved eye detection using MediaPipe Face Mesh
- More robust face landmark processing
- Better thresholds and error handling

Installation requirements:
pip install opencv-python pillow imagehash mediapipe numpy

Usage:
python cull_fixed.py --input ./photos --output ./culled
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import hashlib

import cv2
import numpy as np
from PIL import Image
import imagehash
import mediapipe as mp


class ImprovedPhotoCuller:
    """Improved photo culling with better detection algorithms."""

    def __init__(self, input_dir: str, output_dir: str, sharpness_threshold: float = 50.0):
        """
        Initialize the PhotoCuller with improved defaults.

        Args:
            input_dir: Path to input directory containing images
            output_dir: Path to output directory for sorted images
            sharpness_threshold: Minimum sharpness score (lowered default)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sharpness_threshold = sharpness_threshold

        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

        # Initialize MediaPipe Face Mesh with better settings
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lowered for better detection
            min_tracking_confidence=0.3
        )

        # More precise eye landmark indices for MediaPipe Face Mesh
        # Left eye outer corner, top, inner corner, bottom points
        self.left_eye_key_points = [33, 160, 158, 133, 153, 144]
        # Right eye outer corner, top, inner corner, bottom points
        self.right_eye_key_points = [362, 385, 387, 263, 373, 380]

        # Eye contour for more precise detection
        self.left_eye_contour = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_contour = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Statistics
        self.stats = {
            'total_processed': 0,
            'selected': 0,
            'blurry': 0,
            'closed_eye': 0,
            'duplicates': 0,
            'others': 0
        }

        # Store image hashes for duplicate detection
        self.image_hashes: Dict[str, str] = {}
        self.hash_groups: Dict[str, List[str]] = {}

    def setup_output_directories(self) -> None:
        """Create output directory structure."""
        categories = ['selected', 'blurry', 'closed_eye', 'duplicates', 'others']

        for category in categories:
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {category_dir}")

    def get_image_files(self) -> List[Path]:
        """Get all supported image files from input directory."""
        image_files = []

        for file_path in self.input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                image_files.append(file_path)

        print(f"Found {len(image_files)} image files")
        return image_files

    def calculate_sharpness_multiple_methods(self, image_path: Path) -> Dict[str, float]:
        """
        Calculate image sharpness using multiple methods for better accuracy.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with different sharpness metrics
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'laplacian': 0.0, 'sobel': 0.0, 'tenengrad': 0.0, 'combined': 0.0}

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Method 1: Variance of Laplacian
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Method 2: Sobel gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            sobel_mean = np.mean(sobel_magnitude)

            # Method 3: Tenengrad (squared gradient)
            tenengrad = np.mean(sobelx ** 2 + sobely ** 2)

            # Combined score (weighted average)
            combined_score = (laplacian_var * 0.4 + sobel_mean * 0.3 + tenengrad * 0.0001)

            return {
                'laplacian': float(laplacian_var),
                'sobel': float(sobel_mean),
                'tenengrad': float(tenengrad),
                'combined': float(combined_score)
            }

        except Exception as e:
            print(f"Error calculating sharpness for {image_path}: {e}")
            return {'laplacian': 0.0, 'sobel': 0.0, 'tenengrad': 0.0, 'combined': 0.0}

    def calculate_eye_aspect_ratio_improved(self, eye_points: np.ndarray, image_shape: tuple) -> float:
        """
        Improved Eye Aspect Ratio calculation.

        Args:
            eye_points: Array of eye landmark coordinates (normalized)
            image_shape: Shape of the image (height, width)

        Returns:
            Eye aspect ratio (lower values indicate closed eyes)
        """
        try:
            # Convert normalized coordinates to pixel coordinates
            h, w = image_shape[:2]
            eye_pixels = eye_points.copy()
            eye_pixels[:, 0] *= w
            eye_pixels[:, 1] *= h

            # Calculate EAR using multiple vertical distances
            # Vertical distances
            v1 = np.linalg.norm(eye_pixels[1] - eye_pixels[5])  # Top to bottom
            v2 = np.linalg.norm(eye_pixels[2] - eye_pixels[4])  # Second vertical line

            # Horizontal distance
            h1 = np.linalg.norm(eye_pixels[0] - eye_pixels[3])  # Left to right corner

            # Calculate EAR
            if h1 > 0:
                ear = (v1 + v2) / (2.0 * h1)
            else:
                ear = 0.0

            return ear

        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.25  # Default to open eye value

    def detect_faces_and_eyes_improved(self, image_path: Path) -> Tuple[int, int, int, Dict]:
        """
        Improved face and eye detection with better accuracy.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (num_faces, num_eyes, num_closed_eyes, debug_info)
        """
        debug_info = {'face_detection_confidence': [], 'eye_ratios': []}

        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return 0, 0, 0, debug_info

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # Process with MediaPipe
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return 0, 0, 0, debug_info

            num_faces = len(results.multi_face_landmarks)
            num_eyes = 0
            num_closed_eyes = 0

            # Analyze each face
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks as normalized coordinates
                landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

                # Left eye analysis
                left_eye_points = landmarks[self.left_eye_key_points]
                left_ear = self.calculate_eye_aspect_ratio_improved(left_eye_points, (h, w))
                debug_info['eye_ratios'].append(('left', left_ear))
                num_eyes += 1

                # Adjusted threshold for closed eye detection
                if left_ear < 0.15:  # More sensitive threshold
                    num_closed_eyes += 1

                # Right eye analysis
                right_eye_points = landmarks[self.right_eye_key_points]
                right_ear = self.calculate_eye_aspect_ratio_improved(right_eye_points, (h, w))
                debug_info['eye_ratios'].append(('right', right_ear))
                num_eyes += 1

                if right_ear < 0.15:  # More sensitive threshold
                    num_closed_eyes += 1

            return num_faces, num_eyes, num_closed_eyes, debug_info

        except Exception as e:
            print(f"Error detecting faces/eyes for {image_path}: {e}")
            return 0, 0, 0, debug_info

    def calculate_perceptual_hash(self, image_path: Path) -> str:
        """Calculate perceptual hash for duplicate detection."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Calculate perceptual hash
                hash_value = imagehash.phash(img, hash_size=8)
                return str(hash_value)

        except Exception as e:
            print(f"Error calculating hash for {image_path}: {e}")
            return hashlib.md5(str(image_path).encode()).hexdigest()[:16]

    def find_duplicates(self, image_files: List[Path]) -> Set[str]:
        """Find duplicate images using perceptual hashing."""
        print("Calculating perceptual hashes for duplicate detection...")

        # Calculate hashes for all images
        for image_path in image_files:
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
                        # Convert hex strings to integers for comparison
                        h1_int = int(hash1, 16)
                        h2_int = int(hash2, 16)

                        # Calculate Hamming distance
                        hamming_distance = bin(h1_int ^ h2_int).count('1')

                        if hamming_distance <= 8:  # Threshold for similarity
                            similar_group.extend(paths2)
                            processed_hashes.add(hash2)
                    except ValueError:
                        # Fallback to string comparison
                        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
                        if hamming_distance <= 5:
                            similar_group.extend(paths2)
                            processed_hashes.add(hash2)

            if len(similar_group) > 1:
                duplicate_paths.update(similar_group[1:])

            processed_hashes.add(hash1)

        print(f"Found {len(duplicate_paths)} duplicate images")
        return duplicate_paths

    def classify_image(self, image_path: Path, duplicates: Set[str]) -> Tuple[str, Dict]:
        """
        Classify an image with detailed reasoning.

        Args:
            image_path: Path to the image file
            duplicates: Set of duplicate image paths

        Returns:
            Tuple of (category, debug_info)
        """
        debug_info = {}
        image_path_str = str(image_path)

        # Check if duplicate
        if image_path_str in duplicates:
            debug_info['reason'] = 'Detected as duplicate'
            return 'duplicates', debug_info

        # Calculate sharpness with multiple methods
        sharpness_scores = self.calculate_sharpness_multiple_methods(image_path)
        debug_info['sharpness'] = sharpness_scores

        # Use combined score for classification
        combined_sharpness = sharpness_scores['combined']

        # Check if blurry (adjusted threshold)
        if combined_sharpness < self.sharpness_threshold:
            debug_info['reason'] = f'Low sharpness: {combined_sharpness:.2f} < {self.sharpness_threshold}'
            return 'blurry', debug_info

        # Detect faces and eyes
        num_faces, num_eyes, num_closed_eyes, face_debug = self.detect_faces_and_eyes_improved(image_path)
        debug_info.update(face_debug)
        debug_info['faces'] = num_faces
        debug_info['eyes'] = num_eyes
        debug_info['closed_eyes'] = num_closed_eyes

        # If no faces detected, put in others
        if num_faces == 0:
            debug_info['reason'] = 'No faces detected'
            return 'others', debug_info

        # Check for closed eyes (majority closed)
        if num_eyes > 0:
            closed_ratio = num_closed_eyes / num_eyes
            debug_info['closed_eye_ratio'] = closed_ratio

            if closed_ratio > 0.6:  # More than 60% of eyes closed
                debug_info['reason'] = f'Majority of eyes closed: {closed_ratio:.2%}'
                return 'closed_eye', debug_info

        # If all checks pass, it's selected
        debug_info['reason'] = 'Passed all quality checks'
        return 'selected', debug_info

    def copy_image_to_category(self, image_path: Path, category: str) -> None:
        """Copy image to the appropriate category folder."""
        try:
            dest_dir = self.output_dir / category
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
            print(f"Error copying {image_path} to {category}: {e}")

    def process_images(self, verbose: bool = False) -> None:
        """Main processing function with optional verbose output."""
        print(f"Starting improved photo culling process...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Sharpness threshold: {self.sharpness_threshold}")

        # Setup output directories
        self.setup_output_directories()

        # Get all image files
        image_files = self.get_image_files()
        if not image_files:
            print("No image files found!")
            return

        # Find duplicates
        duplicates = self.find_duplicates(image_files)

        # Process each image
        print(f"\nProcessing {len(image_files)} images...")

        for i, image_path in enumerate(image_files, 1):
            try:
                # Classify image with debug info
                category, debug_info = self.classify_image(image_path, duplicates)

                if verbose:
                    print(f"\n{image_path.name}:")
                    print(f"  Category: {category}")
                    print(f"  Reason: {debug_info.get('reason', 'Unknown')}")
                    if 'sharpness' in debug_info:
                        print(f"  Sharpness (combined): {debug_info['sharpness']['combined']:.2f}")
                    if 'eye_ratios' in debug_info:
                        for eye, ratio in debug_info['eye_ratios']:
                            print(f"  {eye} eye ratio: {ratio:.3f}")

                # Copy to appropriate folder
                self.copy_image_to_category(image_path, category)

                # Update statistics
                self.stats['total_processed'] += 1
                self.stats[category] += 1

                # Progress indicator
                if i % 10 == 0 or i == len(image_files):
                    print(f"Processed {i}/{len(image_files)} images")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                self.copy_image_to_category(image_path, 'others')
                self.stats['total_processed'] += 1
                self.stats['others'] += 1

        print(f"\nPhoto culling completed!")

    def print_summary(self) -> None:
        """Print JSON summary of results."""
        summary = {
            "improved_photo_culling_summary": {
                "input_directory": str(self.input_dir),
                "output_directory": str(self.output_dir),
                "sharpness_threshold": self.sharpness_threshold,
                "statistics": self.stats,
                "percentages": {
                    category: round((count / max(self.stats['total_processed'], 1)) * 100, 1)
                    for category, count in self.stats.items()
                    if category != 'total_processed'
                }
            }
        }

        print("\n" + "=" * 50)
        print("IMPROVED PHOTO CULLING SUMMARY")
        print("=" * 50)
        print(json.dumps(summary, indent=2))


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Improved photo culling with better detection algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing images to process'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for sorted images'
    )

    parser.add_argument(
        '--sharpness-threshold',
        type=float,
        default=50.0,  # Lowered default threshold
        help='Minimum sharpness score (combined metric)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with classification details'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist!")
        return 1

    if not os.path.isdir(args.input):
        print(f"Error: '{args.input}' is not a directory!")
        return 1

    try:
        # Create and run improved photo culler
        culler = ImprovedPhotoCuller(args.input, args.output, args.sharpness_threshold)
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