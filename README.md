# Digital Tape Measure - Wall Measurement System

This project extracts real-world dimensions from perspective-distorted photos of walls using computer vision techniques.

Project Structure

```
.
├── Raw Images/          # Original JPG photos
├── Input/              # Reference images with laser grid
├── Output/             # Ground truth measurements
├── level1.py           # Basic perspective correction
├── level2.py           # With lens distortion handling
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Requires Python 3.7 or higher.

## Usage

### Level 1 - Basic Wall Flattening

Handles simple perspective transformation with clean laser detection.

```bash
python level1.py
```

Output saved to `output_level1/` folder.

### Level 2 - Distortion Correction

Adds lens distortion correction before perspective transform for better accuracy.

```bash
python level2.py
```

Output saved to `output_level2/` folder.

## How It Works

### Level 1 Approach

1. Load input image with red laser cross
2. Convert to HSV color space for red detection
3. Create binary mask for red pixels (handles wraparound at 0/180)
4. Clean mask with morphological operations
5. Detect lines using Canny + Hough transform
6. Separate into horizontal and vertical lines based on angle
7. Find bounding box from line extents
8. Apply perspective transform to raw image using detected corners
9. Calculate pixel-to-mm scale using known wall height
10. Compute wall width from flattened image dimensions

Level 2 Additions

1. Undistort both raw and input images using camera calibration parameters
2. More robust line detection with better parameter tuning
3. Line merging algorithm to handle multiple detected segments
4. Fallback to original image if undistortion causes detection failure
5. Improved scale calculation with higher precision

Assumptions

- Laser lines form a clear cross pattern visible in input images
- Wall height measurements are provided (2710mm for all test images)
- Red laser wavelength is consistent across images
- Wall is approximately planar (small deviations acceptable)
- Camera distortion follows standard radial model
- Lighting conditions allow red color detection in HSV space

Results Format

The `results.json` file contains:

```json
{
  "IMG_7282": {
    "wall_height_mm": 2710.0,
    "wall_width_mm": 3842.15,
    "height_pixels": 1523,
    "width_pixels": 2160,
    "scale_px_per_mm": 0.5621
  }
}
```

Level 3 Approach (Not Implemented)

For handling non-planar walls with surface irregularities:

1. Detect full laser grid instead of just cross endpoints
2. Create dense correspondence between grid points
3. Apply piecewise affine or thin plate spline transformation
4. Use local warping based on grid deformation
5. Requires more sophisticated grid detection (possibly template matching)
6. Would need to segment wall into local regions for independent warping

Could use scikit-image PiecewiseAffineTransform or implement TPS warping with scipy.

Known Limitations

- Red detection may fail in poor lighting
- Assumes relatively clean wall backgrounds
- Camera calibration uses estimated parameters (not measured)
- No automatic measurement extraction from output images
- Doesn't handle cases where laser is partially occluded

## Notes

Wall height measurements were manually extracted from output images. For production use, implement OCR or provide measurements separately.
