import cv2
import numpy as np
import json
import os

def detect_laser_cross(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return None
    
    horizontal = []
    vertical = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
        
        if angle < 20 or angle > 160:
            horizontal.append((x1, y1, x2, y2))
        elif 70 < angle < 110:
            vertical.append((x1, y1, x2, y2))
    
    if not horizontal or not vertical:
        return None
    
    h_xs = [x for line in horizontal for x in [line[0], line[2]]]
    h_ys = [y for line in horizontal for y in [line[1], line[3]]]
    v_xs = [x for line in vertical for x in [line[0], line[2]]]
    v_ys = [y for line in vertical for y in [line[1], line[3]]]
    
    x_min = min(min(h_xs), min(v_xs))
    x_max = max(max(h_xs), max(v_xs))
    y_min = min(min(h_ys), min(v_ys))
    y_max = max(max(h_ys), max(v_ys))
    
    margin = 30
    corners = np.array([
        [x_min - margin, y_min - margin],
        [x_max + margin, y_min - margin],
        [x_max + margin, y_max + margin],
        [x_min - margin, y_max + margin]
    ], dtype=np.float32)
    
    return corners

def flatten_wall(raw_img, corners):
    width = int(max(
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[2] - corners[3])
    ))
    height = int(max(
        np.linalg.norm(corners[0] - corners[3]),
        np.linalg.norm(corners[1] - corners[2])
    ))
    
    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(raw_img, M, (width, height))
    
    return warped

def process_images(raw_folder, input_folder, output_folder, image_mapping, measurements):
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    for input_file in input_files:
        input_base = input_file.replace('.jpg', '')
        
        if input_base not in image_mapping:
            print(f"No mapping for {input_file}")
            continue
        
        raw_base = image_mapping[input_base]
        raw_file = raw_base + '.JPG'
        
        input_path = os.path.join(input_folder, input_file)
        raw_path = os.path.join(raw_folder, raw_file)
        
        if not os.path.exists(raw_path):
            print(f"Skip {input_file} - raw image {raw_file} not found")
            continue
        
        print(f"Processing {input_file} -> {raw_file}")
        
        input_img = cv2.imread(input_path)
        raw_img = cv2.imread(raw_path)
        
        if input_img is None or raw_img is None:
            print(f"Failed to load images")
            continue
        
        corners = detect_laser_cross(input_img)
        
        if corners is None:
            print(f"Failed to detect laser cross")
            continue
        
        warped = flatten_wall(raw_img, corners)
        
        output_path = os.path.join(output_folder, f"{raw_base}_flattened.jpg")
        cv2.imwrite(output_path, warped)
        
        if input_base in measurements:
            known_height_mm = measurements[input_base]
            height_px = warped.shape[0]
            width_px = warped.shape[1]
            
            px_per_mm = height_px / known_height_mm
            width_mm = width_px / px_per_mm
            
            results[raw_base] = {
                "wall_height_mm": round(known_height_mm, 2),
                "wall_width_mm": round(width_mm, 2),
                "height_pixels": height_px,
                "width_pixels": width_px
            }
            
            print(f"  Height: {known_height_mm}mm, Width: {width_mm:.2f}mm")
    
    with open(os.path.join(output_folder, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Check {output_folder}/results.json")

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
    
    measurements = {
        "IMG_7282": 2710,
        "IMG_7284": 2710,
        "IMG_7286": 2710,
        "IMG_7288": 2710,
        "IMG_7290": 2710,
        "IMG_7292": 2710,
        "IMG_7294": 2710,
        "IMG_7296": 2710
    }
    
    process_images(
        raw_folder="Raw Images",
        input_folder="Input",
        output_folder="output_level1",
        image_mapping=image_mapping,
        measurements=measurements
    )