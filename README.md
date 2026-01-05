# Hades II Icon Detector

An automated computer vision tool that extracts run data from *Hades II* victory screens. It uses YOLOv8 for grid detection and advanced Template Matching to identify Boons, Aspects, Keepsakes, and Familiars with high accuracy—even when pinned or ranked.

## Features
* **Auto-Detection:** Identifies the Aspect, Familiar, and all Boons/Upgrades from a screenshot.
* **Stat Extraction:** Reads "Clear Time" and "Fear" levels using OCR.
* **Smart Matching:** Handles "Pinned" icons, Ranks, and text overlays using multi-zone template matching.
* **CSV Export:** Outputs all run data into a clean `hades_run_data.csv`.

## 📦 Setup

1.  **Install Dependencies**
    ```bash
    pip install opencv-python numpy pandas easyocr ultralytics
    ```

2.  **Download Assets**
    This tool requires a specific set of reference images and models to work.
    * **Reference Database:** [Download Here](https://drive.google.com/uc?export=download&id=1nKrHEYRW5VM06OpFXZVCWk4zZtxYDGKB)

3.  **Project Structure**
    Extract the downloaded assets and ensure your folder looks exactly like this:
    ```text
    Project_Folder/
    ├── main.py                # The script
    ├── vic_screens/           # PUT YOUR SCREENSHOTS HERE
    └── assets/                # EXTRACTED ASSETS GO HERE
        ├── best.pt            # YOLO Model
        ├── iconname_code_display.csv
        ├── pin_overlay.png
        ├── rank_1.png ... rank_4.png
        └── reference_icons/   # Folder containing icon subfolders
    ```

## 🎮 Usage

**1. Standard Run (Fast)**
Processes all images in `vic_screens/` and saves data to CSV.
```bash
python main.py