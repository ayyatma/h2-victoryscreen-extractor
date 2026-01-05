import cv2
import numpy as np
import os
import glob

# --- INPUT / OUTPUT ---
INPUT_DIR = "vic_screens"
OUTPUT_DIR = "debug_labels_filtered" # New folder to compare

GRID_CONFIG = {
    "start_x": 10, "start_y": 198, 
    "icon_size": 80, 
    "gap_x": 95, "gap_y": 90, 
    "rows": 7, "cols": 5
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_slot_filled(crop):
    """
    Returns True if the slot has an icon (High visual variance).
    Returns False if it's empty (Flat dark color).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Variance Check: Icons have high detail (variance > 500 usually)
    # Empty slots are flat (variance < 100 usually)
    variance = np.var(gray)
    
    # 2. Brightness Check: Icons are brighter than the background
    mean_brightness = np.mean(gray)
    
    # Thresholds: Adjust these if it misses dark icons or grabs empty ones
    return variance > 150 and mean_brightness > 20

def main():
    search_path = os.path.join(INPUT_DIR, "*")
    files = glob.glob(search_path)
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    raw_files = [f for f in files if f.lower().endswith(valid_exts)]
    
    if not raw_files:
        print(f"❌ No images found in '{INPUT_DIR}/'")
        return

    print(f"👀 Visualizing FILTERED labels for {len(raw_files)} images...")

    for f in raw_files:
        img = cv2.imread(f)
        if img is None: continue
        
        # Resize to 1080p standard
        img_resized = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        
        for r in range(GRID_CONFIG["rows"]):
            for c in range(GRID_CONFIG["cols"]):
                x = GRID_CONFIG["start_x"] + (c * GRID_CONFIG["gap_x"])
                y = GRID_CONFIG["start_y"] + (r * GRID_CONFIG["gap_y"])
                w = GRID_CONFIG["icon_size"]
                h = GRID_CONFIG["icon_size"]
                
                # CROP the slot to check it
                crop = img_resized[y:y+h, x:x+w]
                
                if is_slot_filled(crop):
                    # Draw Green Box (It's a Boon!)
                    cv2.rectangle(img_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    # Optional: Draw Red X for empty slots (just for visual confirmation)
                    # cv2.line(img_resized, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    # cv2.line(img_resized, (x+w, y), (x, y+h), (0, 0, 255), 1)
                    pass

        base_name = os.path.basename(f)
        out_path = os.path.join(OUTPUT_DIR, f"viz_{base_name}")
        cv2.imwrite(out_path, img_resized)
        
    print(f"✅ Done! Check '{OUTPUT_DIR}'. Only REAL boons should have boxes now.")

if __name__ == "__main__":
    main()