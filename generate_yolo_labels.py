import cv2
import numpy as np
import os
import glob

# --- INPUT / OUTPUT ---
INPUT_DIR = "vic_screens"
DATASET_DIR = "datasets/hades_grid"

IMG_OUTPUT_DIR = os.path.join(DATASET_DIR, "images")
LABEL_OUTPUT_DIR = os.path.join(DATASET_DIR, "labels")

GRID_CONFIG = {
    "start_x": 10, "start_y": 198, 
    "icon_size": 80, 
    "gap_x": 95, "gap_y": 90, 
    "rows": 7, "cols": 5
}

os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)

def is_slot_filled(crop):
    """ Same logic as the visualizer """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    mean_brightness = np.mean(gray)
    return variance > 150 and mean_brightness > 20

def convert_to_yolo(x, y, w, h, img_w, img_h):
    center_x = (x + (w / 2)) / img_w
    center_y = (y + (h / 2)) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"

def main():
    search_path = os.path.join(INPUT_DIR, "*")
    files = glob.glob(search_path)
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    raw_files = [f for f in files if f.lower().endswith(valid_exts)]
    
    if not raw_files:
        print(f"❌ No images found in '{INPUT_DIR}/'.")
        return

    print(f"🚀 Generating CLEAN dataset from {len(raw_files)} images...")

    for f in raw_files:
        img = cv2.imread(f)
        if img is None: continue

        # Resize
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        
        # Save image
        base_name = os.path.splitext(os.path.basename(f))[0]
        out_img_name = f"{base_name}.jpg"
        cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, out_img_name), img)
        
        # Save Label
        label_path = os.path.join(LABEL_OUTPUT_DIR, f"{base_name}.txt")
        
        with open(label_path, "w") as label_file:
            for r in range(GRID_CONFIG["rows"]):
                for c in range(GRID_CONFIG["cols"]):
                    x = GRID_CONFIG["start_x"] + (c * GRID_CONFIG["gap_x"])
                    y = GRID_CONFIG["start_y"] + (r * GRID_CONFIG["gap_y"])
                    w = GRID_CONFIG["icon_size"]
                    h = GRID_CONFIG["icon_size"]
                    
                    # CHECK IF FILLED
                    crop = img[y:y+h, x:x+w]
                    if is_slot_filled(crop):
                        yolo_line = convert_to_yolo(x, y, w, h, 1920, 1080)
                        label_file.write(yolo_line + "\n")
                    
    print(f"✅ Clean dataset generated in {DATASET_DIR}")
    
    # Write Config
    yaml_content = f"""
path: {os.path.abspath(DATASET_DIR)}
train: images
val: images
names:
  0: boon_slot
"""
    with open("hades_grid.yaml", "w") as f:
        f.write(yaml_content)
    print("✅ Created 'hades_grid.yaml'")

if __name__ == "__main__":
    main()