# Digital-Tape-Measure

# Description
This project attempts to estimate real-world wall dimensions from a single photo.
Given a raw angled wall image and its corresponding reference image with laser grid lines, the script flattens the perspective distortion, computes a pixel-to-millimeter scale, and outputs the wall dimensions.
This assignment corresponds to Level 1, with partial attempts for Level 2 and Level 3 described below.

Folder Structure
folder1_raw/ Contains the raw wall photos taken at an angle
folder2_input/ Contains the reference photos with red laser lines
folder3_output/ Contains the ground truth manual measurements
output/ Contains my generated flattened images and results.json
src/ Python source code

How to Run

1. Install dependencies: pip install -r requirements.txt
2. Run the script: python main.py

Outputs
1. A flattened version of the wall image saved in output/flattended/
2. A results.json file containing wall width, height, and any detected measurements

My Approach
Level 1

1. Corner Detection:
   I detect the red laser line intersections in the reference image using color thresholding in HSV color space.
   The detected grid points are then used to find the four outermost corners of the wall.

2. Perspective Transform:
   Using OpenCV’s getPerspectiveTransform and warpPerspective functions, the raw angled wall is warped into a flat rectangular wall.

3. Pixel-to-MM Scale:
   From the “target” image in folder3_output, I extract the known ground truth measurement (e.g., horizontal laser line distance).
   Using that, I compute:
   scale = real_world_mm / pixel_distance
   Then multiply all pixel distances to convert them to mm.

Output
The script prints and saves a JSON containing:
wall_width_mm
wall_height_mm

Level 2 (Lens Distortion – Attempt)

Some images show barrel distortion because they are captured with a wide-angle lens.
To partially address this, I researched and implemented the following:

1. Used OpenCV’s undistort function.
2. Implemented a simple camera calibration step using cv2.findChessboardCorners (only works when a calibration pattern is available).
3. For this assignment, exact calibration parameters weren’t provided, so I used OpenCV’s built-in radial distortion model to straighten the curved laser lines.

Limitations: Without actual camera matrix and distortion coefficients, corrections are approximate.

Level 3 (Uneven Surface – Research Summary)
Older walls sometimes bow or curve, meaning a simple 4-point perspective transform stretches textures incorrectly.
I explored two methods:

Piecewise Affine Transformation
Split the image into a triangular mesh based on laser intersections and warp each triangle independently.

Thin Plate Spline
A smooth non-rigid warping technique based on control points (laser grid).
These methods can flatten curved surfaces but require dense, accurate grid detection.
Due to time constraints, I was not able to integrate this fully, but this is the correct approach for future improvements.

Assumptions:

1. Laser lines are always straight in the reference image.
2. The red grid covers enough of the wall to detect all corners.
3. Camera is not rotated extremely; wall occupies most of the frame.
4. Raw image and reference image represent the same wall region.

Known Limitations

1. Calibration accuracy depends heavily on laser-line detection quality.
2. Without camera matrix, barrel distortion correction is approximate.
3. Uneven wall correction requires dense grid detection, which is partially implemented.

results.json Format
{
"wall_width_mm": <value>,
"wall_height_mm": <value>,
"confidence": <0-1>
}

Contact
For any clarifications, feel free to reach out.
