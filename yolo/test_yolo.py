from ultralytics import YOLO
import cv2
import glob
import os

MODEL_PATH = 'runs/detect/train/weights/best.pt'
INPUT_FOLDER = "../vic_screens"
OUTPUT_FOLDER = "debug_yolo_clean"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    files.sort()
    
    # Test first 10
    test_files = files[:10]
    
    print(f"🧠 Testing model on {len(test_files)} images (Clean Mode)...")

    for f in test_files:
        # Run inference
        # verbose=False stops it from printing "15 boon_slots..." to console
        results = model(f, verbose=False) 
        
        # Load the original image to draw on
        img = cv2.imread(f)
        
        # Iterate through detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Draw simple Green Box (Thickness = 2)
                # No text, no confidence score
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Save
        base_name = os.path.basename(f)
        out_path = os.path.join(OUTPUT_FOLDER, f"clean_{base_name}")
        cv2.imwrite(out_path, img)
        print(f"   Saved: {out_path}")
        
    print(f"✅ Done! Check '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    main()