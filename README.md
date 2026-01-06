# Hades II Icon Detector

An automated computer vision tool that extracts run data from *Hades II* victory screens. It uses YOLOv8 for grid detection, advanced Template Matching for icon identification, and Fuzzy OCR for robust stat extraction.

## Features
* **Auto-Detection:** Identifies Aspects, Familiars, and Boons using multi-zone template matching.
* **Robust OCR:** Extracts "Clear Time" and "Fear" using **Smart Binarization** and **Fuzzy String Matching** to handle typos and low contrast.
* **Dynamic Variant Generation:** Automatically generates "Pinned" and "Ranked" variants of reference icons to ensure accurate matching.
* **Spatial Parsing:** Locates stat values relative to their labels dynamically, handling shifting UI elements.
* **CSV Export:** Outputs all run data into a clean `hades_run_data.csv`.

## 📦 Setup

1.  **Install Dependencies**
    ```bash
    pip install opencv-python numpy pandas easyocr ultralytics
    ```

2.  **Download Assets**
    * **Reference Database:** [Download Here](https://drive.google.com/uc?export=download&id=1nKrHEYRW5VM06OpFXZVCWk4zZtxYDGKB)

3.  **Project Structure**
    Ensure your folder looks exactly like this:
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

Run the script from your terminal:

```bash
python main.py [options]
```

### Options
| Flag | Description |
| :--- | :--- |
| **None** | Standard run. Processes all images in `vic_screens/` and saves to CSV. |
| `-d`, `--debug` | **Debug Mode.** Saves visual overlays (grid/OCR boxes) and individual icon crops to `debug_output/`. |
| `--gpu` | **GPU Acceleration.** Forces YOLO and EasyOCR to use CUDA (if available) for faster processing. |

### Examples
**Standard Run:**
```bash
python main.py
```

**Run with Debug Visuals:**
```bash
python main.py --debug
```

**Run on GPU:**
```bash
python main.py --gpu
```