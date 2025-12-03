import cv2
import numpy as np
import json
import os

def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)
    
    x, y, w, h = roi
    if h > 0 and w > 0:
        undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def detect_red_lines_multi_method(image):
    methods = [
        lambda img: detect_method_1(img),
        lambda img: detect_method_2(img),
        lambda img: detect_method_3(img)
    ]
    
    for method in methods:
        corners = method(image)
        if corners is not None:
            return corners
    
    return None

def detect_method_1(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 50, 50])
    upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=80, maxLineGap=20)
    
    return extract_corners_from_lines(lines, image.shape)

def detect_method_2(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    red_mask = cv2.inRange(a, 130, 255)
    
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    edges = cv2.Canny(red_mask, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=60, maxLineGap=25)
    
    return extract_corners_from_lines(lines, image.shape)

def detect_method_3(image):
    b, g, r = cv2.split(image)
    
    red_enhanced = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
    _, red_mask = cv2.threshold(red_enhanced, 30, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    edges = cv2.Canny(red_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=30)
    
    return extract_corners_from_lines(lines, image.shape)

def extract_corners_from_lines(lines, shape):
    if lines is None or len(lines) < 2:
        return None
    
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if length < 40:
            continue
        
        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
        
        if angle < 30 or angle > 150:
            h_lines.append(line[0])
        elif 60 < angle < 120:
            v_lines.append(line[0])
    
    if not h_lines or not v_lines:
        return None
    
    def get_line_bounds(lines, is_horizontal):
        lines = np.array(lines)
        
        if is_horizontal:
            all_x = np.concatenate([lines[:, [0]], lines[:, [2]]])
            all_y = np.concatenate([lines[:, [1]], lines[:, [3]]])
            x_min, x_max = all_x.min(), all_x.max()
            y_avg = int(np.median(all_y))
            return [x_min, y_avg, x_max, y_avg]
        else:
            all_x = np.concatenate([lines[:, [0]], lines[:, [2]]])
            all_y = np.concatenate([lines[:, [1]], lines[:, [3]]])
            x_avg = int(np.median(all_x))
            y_min, y_max = all_y.min(), all_y.max()
            return [x_avg, y_min, x_avg, y_max]
    
    h_line = get_line_bounds(h_lines, True)
    v_line = get_line_bounds(v_lines, False)
    
    x_coords = [h_line[0], h_line[2], v_line[0], v_line[2]]
    y_coords = [h_line[1], h_line[3], v_line[1], v_line[3]]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    height, width = shape[:2]
    pad = 50
    
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(width, x_max + pad)
    y_max = min(height, y_max + pad)
    
    corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ], dtype=np.float32)
    
    return corners

def perspective_transform(image, src_corners):
    w1 = np.linalg.norm(src_corners[0] - src_corners[1])
    w2 = np.linalg.norm(src_corners[2] - src_corners[3])
    h1 = np.linalg.norm(src_corners[0] - src_corners[3])
    h2 = np.linalg.norm(src_corners[1] - src_corners[2])
    
    width = int(max(w1, w2))
    height = int(max(h1, h2))
    
    dst_corners = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    result = cv2.warpPerspective(image, matrix, (width, height))
    
    return result

def process_with_distortion_correction(raw_folder, input_folder, output_folder, image_mapping, measurements):
    os.makedirs(output_folder, exist_ok=True)
    
    camera_matrix = np.array([
        [2000, 0, 1500],
        [0, 2000, 2000],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.array([-0.2, 0.1, 0, 0, 0], dtype=np.float32)
    
    results = {}
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    for input_file in input_files:
        input_base = input_file.replace('.jpg', '')
        
        if input_base not in image_mapping:
            print(f"No mapping for {input_file}")
            continue
        
        raw_base = image_mapping[input_base]
        raw_file = raw_base + '.JPG'
        
        raw_path = os.path.join(raw_folder, raw_file)
        input_path = os.path.join(input_folder, input_file)
        
        if not os.path.exists(raw_path):
            print(f"Skip {input_file} - no raw image {raw_file}")
            continue
        
        print(f"Processing {input_file} -> {raw_file}")
        
        input_img = cv2.imread(input_path)
        raw_img = cv2.imread(raw_path)
        
        if input_img is None or raw_img is None:
            print(f"  Failed to load images")
            continue
        
        corners = detect_red_lines_multi_method(input_img)
        
        if corners is None:
            print(f"  Failed - trying undistorted version")
            input_undist = undistort_image(input_img, camera_matrix, dist_coeffs)
            corners = detect_red_lines_multi_method(input_undist)
            
            if corners is None:
                print(f"  Failed for {input_file}")
                continue
            
            raw_img = undistort_image(raw_img, camera_matrix, dist_coeffs)
        
        flattened = perspective_transform(raw_img, corners)
        
        out_path = os.path.join(output_folder, f"{raw_base}_flattened.jpg")
        cv2.imwrite(out_path, flattened)
        
        if input_base in measurements:
            h_mm = measurements[input_base]
            h_px = flattened.shape[0]
            w_px = flattened.shape[1]
            
            scale = h_px / h_mm
            w_mm = w_px / scale
            
            results[raw_base] = {
                "wall_height_mm": round(h_mm, 2),
                "wall_width_mm": round(w_mm, 2),
                "height_pixels": h_px,
                "width_pixels": w_px,
                "scale_px_per_mm": round(scale, 4)
            }
            
            print(f"  Success! H: {h_mm}mm, W: {w_mm:.1f}mm")
        else:
            print(f"  Processed (no measurement data)")
    
    with open(os.path.join(output_folder, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Check {output_folder}/results.json")
    print(f"Successfully processed {len(results)} out of {len(input_files)} images")

if __name__ == "__main__":
    image_mapping = {
        "IMG_7282": "IMG_7281",
        "IMG_7284": "IMG_7283",
        "IMG_7286": "IMG_7285",
        "IMG_7288": "IMG_7287",
        "IMG_7290": "IMG_7289",
        "IMG_7292": "IMG_7291",
        "IMG_7294": "IMG_7293",
        "IMG_7296": "IMG_7295"
    }
    
    wall_heights = {
        "IMG_7282": 2710,
        "IMG_7284": 2710,
        "IMG_7286": 2710,
        "IMG_7288": 2710,
        "IMG_7290": 2710,
        "IMG_7292": 2710,
        "IMG_7294": 2710,
        "IMG_7296": 2710
    }
    
    process_with_distortion_correction(
        raw_folder="Raw Images",
        input_folder="Input",
        output_folder="output_level2",
        image_mapping=image_mapping,
        measurements=wall_heights
    )