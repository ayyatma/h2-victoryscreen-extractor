import cv2
import numpy as np
import glob
import os
import shutil
import pandas as pd
import easyocr
import re
import argparse
from ultralytics import YOLO

# ==========================================
#        1. CENTRAL CONFIGURATION
# ==========================================

# --- DIRECTORIES ---
ASSETS_DIR = "assets"  # <--- Put model, overlays, csv, and icons here
INPUT_DIR = "vic_screens"
DEBUG_DIR = "debug_output"
DEBUG_MODE = False  # Set to True to enable debug visuals

# --- FILE PATHS (Relative to ASSETS_DIR) ---
MODEL_FILE = "best.pt"
MAPPING_FILE = "iconname_code_display.csv"
PIN_OVERLAY_FILE = "pin_overlay.png"
RANK_FILES = ["rank_1.png", "rank_2.png", "rank_3.png", "rank_4.png"]
REF_ICONS_SUBDIR = "reference_icons"
OUTPUT_CSV = "hades_run_data.csv" # Saved to root

# --- IMAGE PROCESSING CONSTANTS ---
TARGET_ICON_SIZE = (90, 90)  # Standard size for reference icons
MATCH_SCENE_WIDTH = int(TARGET_ICON_SIZE[0] * 1.15)      # Width to resize game crop to before matching to allow alignment errors
MATCH_SCALES = [0.7, 0.8, 0.9, 1.0, 1.1] # Scales to test during template matching

# --- MATCHING REGIONS (y_start, y_end, x_start, x_end) ---
# These coordinates are relative to the TARGET_ICON_SIZE (90x90)
TEMPLATE_REGIONS = {
    "head":  (15, 57, 30, 75), # Green: Safe Top-Right (Avoids Pin)
    "heart": (25, 57, 20, 80), # Blue: Safe Center (Avoids Text & Pin)
    "broad": (15, 80, 15, 85)  # Red:  Risky Full Body (Catches everything)
}

# --- THRESHOLDS ---
THRESH_YOLO_CONF = 0.4       # Minimum confidence to detect a slot
THRESH_TEMPLATE_MIN = 0.4    # Minimum score to even consider a candidate
THRESH_SCORE_HIGH = 0.70     # Instant win score
THRESH_SCORE_MED = 0.65      # Win score IF gap is big enough
THRESH_GAP_MIN = 0.05        # Minimum gap required for medium scores

# --- LOGIC CONSTANTS ---
COL0_STRUCTURE = { 0: "aspects", 1: "familiars", 2: "attacks", 3: "specials", 4: "casts", 5: "sprints", 6: "gains" }
COL0_DEFAULTS = { "aspects": "Aspect", "familiars": "Familiar", "attacks": "Attack", "specials": "Special", "casts": "Cast", "sprints": "Sprint", "gains": "Magick" }

# --- REGEX (Compiled for Speed) ---
RE_PINNED = re.compile(r'_PINNED')
RE_RANK = re.compile(r'_RANK_\d+')
RE_CLEAN_DEBUG = re.compile(r'[^\w\-\_\. ]')


# ==========================================
#           2. HELPER FUNCTIONS
# ==========================================

def get_asset_path(filename):
    """Helper to construct paths relative to ASSETS_DIR"""
    return os.path.join(ASSETS_DIR, filename)

def clean_name_internal(internal_name):
    """Removes _PINNED and _RANK suffixes to get the base key."""
    name = RE_PINNED.sub('', internal_name)
    name = RE_RANK.sub('', name)
    return name

def get_display_name(internal_name, mapping):
    """Returns the human-readable name from CSV mapping."""
    if internal_name == "unknown": return "UNKNOWN"
    base_key = clean_name_internal(internal_name)
    return mapping.get(base_key, base_key)

def overlay_image_alpha(img, overlay, opacity=1.0):
    """Blends an overlay (like a pin) onto an icon with alpha transparency."""
    h, w = img.shape[:2]
    if overlay.shape[:2] != (h, w): 
        overlay = cv2.resize(overlay, (w, h))
        
    if img.shape[2] == 4:
        img_bgr, img_alpha = img[:, :, :3], img[:, :, 3] / 255.0
    else:
        img_bgr, img_alpha = img, np.ones((h, w))
    
    if overlay.shape[2] == 4:
        ov_bgr = overlay[:, :, :3]
        ov_alpha = (overlay[:, :, 3] / 255.0) * opacity
    else: return img 

    out_alpha = ov_alpha + img_alpha * (1 - ov_alpha)
    safe_alpha = np.maximum(out_alpha, 1e-6)
    out_bgr = (ov_bgr * ov_alpha[:, :, None] + img_bgr * img_alpha[:, :, None] * (1 - ov_alpha)[:, :, None]) / safe_alpha[:, :, None]
    
    return cv2.merge([
        np.clip(out_bgr, 0, 255).astype(np.uint8)[:,:,i] for i in range(3)
    ] + [np.clip(out_alpha * 255, 0, 255).astype(np.uint8)])


# ==========================================
#           3. RESOURCE LOADING
# ==========================================

def ensure_variants_exist():
    """Generates _PINNED.png and _RANK_X.png variants in the assets folder."""
    print("🔨 Checking/Generating Variants...")
    pin_path = get_asset_path(PIN_OVERLAY_FILE)
    
    if not os.path.exists(pin_path): 
        print(f"⚠️ Pin overlay not found at {pin_path}. Skipping variants.")
        return
    
    pin_overlay = cv2.imread(pin_path, cv2.IMREAD_UNCHANGED)
    rank_overlays = {}
    for r_file in RANK_FILES:
        path = get_asset_path(r_file)
        if os.path.exists(path):
            key = r_file.replace("rank_", "").replace(".png", "")
            rank_overlays[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    ref_base = get_asset_path(REF_ICONS_SUBDIR)
    categories = list(COL0_STRUCTURE.values()) + ["boons", "keepsakes"]
    count = 0

    for cat in categories:
        folder = os.path.join(ref_base, cat)
        if not os.path.exists(folder): continue
        
        is_keepsake = (cat == "keepsakes")
        files = [f for f in glob.glob(os.path.join(folder, "*")) 
                 if not RE_PINNED.search(f) and not RE_RANK.search(f) 
                 and f.lower().endswith(('.png', '.jpg'))]

        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            path = os.path.dirname(f)
            
            if os.path.exists(os.path.join(path, f"{name}_PINNED.png")): continue 
            
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None: continue
            
            # Generate Pinned (0.3 opacity)
            cv2.imwrite(os.path.join(path, f"{name}_PINNED.png"), overlay_image_alpha(img, pin_overlay, 0.3))
            count += 1
            
            # Generate Ranks
            if is_keepsake and rank_overlays:
                for r_key, r_img in rank_overlays.items():
                    ranked = overlay_image_alpha(img, r_img, 1.0)
                    cv2.imwrite(os.path.join(path, f"{name}_RANK_{r_key}.png"), ranked)
                    cv2.imwrite(os.path.join(path, f"{name}_PINNED_RANK_{r_key}.png"), 
                                overlay_image_alpha(ranked, pin_overlay, 0.3))
                    count += 2
                    
    if count: print(f"✨ Generated {count} new variants.")

def load_icon_folder(path):
    d = {}
    if not os.path.exists(path): return d
    for f in glob.glob(os.path.join(path, "*")):
        if f.lower().endswith(('.png','.jpg','.jpeg')):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None: continue
            
            if img.shape[2] == 4:
                coords = cv2.findNonZero(img[:, :, 3])
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    img = img[y:y+h, x:x+w]
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            d[os.path.splitext(os.path.basename(f))[0]] = cv2.resize(img_gray, TARGET_ICON_SIZE)
    return d

def load_resources():
    print("⏳ Loading AI Models & Resources...")
    
    # Check ASSETS_DIR
    if not os.path.exists(ASSETS_DIR):
        print(f"❌ ERROR: Assets directory '{ASSETS_DIR}' not found!")
        exit()

    # Load YOLO
    model_path = get_asset_path(MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        exit()
    grid_model = YOLO(model_path)
    
    # Load OCR
    text_reader = easyocr.Reader(['en'], gpu=True)
    
    # Load Mapping
    mapping = {}
    mapping_path = get_asset_path(MAPPING_FILE)
    if os.path.exists(mapping_path):
        df = pd.read_csv(mapping_path)
        mapping = {str(row['icon_filename']).replace('.png', ''): str(row['display_text']) 
                   for _, row in df.iterrows()}
    else:
        print(f"⚠️ Mapping file not found at {mapping_path}. Using raw filenames.")

    # Load Icons
    libs = {}
    master = {}
    ref_base = get_asset_path(REF_ICONS_SUBDIR)
    
    for v in COL0_STRUCTURE.values():
        libs[v] = load_icon_folder(os.path.join(ref_base, v))
        master.update(libs[v])
        
    libs["boons"] = load_icon_folder(os.path.join(ref_base, "boons"))
    libs["keepsakes"] = load_icon_folder(os.path.join(ref_base, "keepsakes"))
    master.update(libs["boons"])
    master.update(libs["keepsakes"])
    
    # Optimized Overflow Library
    libs["overflow_merged"] = {**libs["boons"], **libs["keepsakes"]}
    
    print(f"✅ Loaded {len(master)} icons.")
    return grid_model, text_reader, libs, master, mapping


# ==========================================
#           4. CORE LOGIC
# ==========================================

def match_icon_specific(crop_img, library, master_lib, mapping, debug_label):
    if not library: return "unknown"
    
    scene = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    if scene.shape[0] > (scene.shape[1] * 1.15): scene = scene[:scene.shape[1], :] 
    
    scale_factor = MATCH_SCENE_WIDTH / scene.shape[1]
    scene = cv2.resize(scene, (MATCH_SCENE_WIDTH, int(scene.shape[0] * scale_factor)))
    
    candidates = [] 
    
    for name, ref in library.items():
        best_tmpl_score = 0.0
        
        # Iterate over Dynamic Configured Regions
        for region_name, (y1, y2, x1, x2) in TEMPLATE_REGIONS.items():
            tmpl = ref[y1:y2, x1:x2]
            
            for s in MATCH_SCALES:
                th, tw = tmpl.shape
                cw, ch = int(tw*s), int(th*s)
                if cw >= scene.shape[1] or ch >= scene.shape[0]: continue
                
                res = cv2.matchTemplate(scene, cv2.resize(tmpl, (cw, ch)), cv2.TM_CCOEFF_NORMED)
                best_tmpl_score = max(best_tmpl_score, np.max(res))
        
        if best_tmpl_score > THRESH_TEMPLATE_MIN: 
            candidates.append((best_tmpl_score, name))

    if not candidates: return "unknown"
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name = candidates[0]
    
    winner_base = clean_name_internal(best_name)
    second_score = 0.0
    for score, name in candidates[1:]:
        if clean_name_internal(name) != winner_base:
            second_score = score
            break
    gap = best_score - second_score

    if best_score > THRESH_SCORE_HIGH or (best_score > THRESH_SCORE_MED and gap > THRESH_GAP_MIN): 
        return best_name
    
    print(f"⚠️  [DEBUG] '{debug_label}' Unmatched (Best: {best_score:.3f}, Gap: {gap:.3f}). Candidates:")
    for s, n in candidates[:3]: 
        print(f"      - {get_display_name(n, mapping)}: {s:.3f}")
    
    if master_lib: 
        return match_icon_specific(crop_img, master_lib, None, mapping, debug_label + " (FB)")
        
    return "unknown"

def analyze_grid_structure(slots, img_h):
    if not slots: return [], {}
    
    avg_w = np.mean([s[2] for s in slots])
    avg_h = np.mean([s[3] for s in slots])
    
    slots_wc = sorted([{'bbox': s, 'cx': s[0]+s[2]/2, 'cy': s[1]+s[3]/2} for s in slots], key=lambda s: s['cx'])
    cols, cur = [], [slots_wc[0]]
    for i in range(1, len(slots_wc)):
        if (slots_wc[i]['cx'] - slots_wc[i-1]['cx']) > (avg_w * 0.8):
            cols.append(cur); cur = [slots_wc[i]]
        else: cur.append(slots_wc[i])
    cols.append(cur)
    
    gaps = []
    for c in cols:
        if len(c) < 2: continue
        c.sort(key=lambda s: s['cy'])
        for k in range(len(c)-1):
            d = c[k+1]['cy'] - c[k]['cy']
            if (avg_h * 0.9) < d < (avg_h * 1.4): gaps.append(d)
    
    row_unit = np.median(gaps) if gaps else avg_h * 1.15
    anchor_y = sorted(cols[0], key=lambda s: s['cy'])[0]['cy']
    
    final, c0_occ = [], {}
    for idx, col in enumerate(cols):
        for itm in col:
            r = int(round((itm['cy'] - anchor_y) / row_unit))
            if idx == 0 and 0 <= r <= 6:
                ent = {'bbox': itm['bbox'], 'col': 0, 'row': r, 'score': abs((itm['cy']-anchor_y)-(r*row_unit)), 'cc': 0}
                if r in c0_occ:
                    if ent['score'] < c0_occ[r]['score']: 
                        final.append({**c0_occ[r], 'col': 1})
                        c0_occ[r] = ent
                    else: 
                        final.append({**ent, 'col': 1})
                else: c0_occ[r] = ent
            else: 
                final.append({'bbox': itm['bbox'], 'col': 1, 'row': r, 'cc': idx})
                
    final.extend(c0_occ.values())
    return sorted(final, key=lambda s: (s['col'], s['row'], s.get('cc', 0))), {
        'col_centers': [np.mean([i['cx'] for i in c]) for c in cols], 
        'row_unit': row_unit, 
        'anchor_cy': anchor_y
    }

def get_stats(reader, img):
    h, w = img.shape[:2]
    stats = {'Clear Time': 'Unknown', 'Fear': '0'}
    crop = img[int(h*0.15):int(h*0.35), int(w*0.75):w]
    res = reader.readtext(crop, detail=0, allowlist='0123456789:.,UsedFar ')
    text_blob = "".join(res)
    tm = re.search(r'(\d{1,2})[:.,](\d{2})', text_blob)
    if tm: stats['Clear Time'] = f"{tm.group(1)}:{tm.group(2)}"
    for t in res:
        fm = re.search(r'Used.*?(\d+)', t)
        if fm: stats['Fear'] = fm.group(1)
    return stats


# ==========================================
#         5. VISUALIZATION & DEBUG
# ==========================================

def setup_debug_folder():
    if DEBUG_MODE:
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        os.makedirs(os.path.join(DEBUG_DIR, "crops"))

def visualize_grid_logic(img, slots, debug_info, filename, slot_labels=None, save=True):
    vis = img.copy()
    ref_w, ref_h = TARGET_ICON_SIZE

    # Draw Grid
    for col_x in debug_info.get('col_centers', []):
        cv2.line(vis, (int(col_x), 0), (int(col_x), img.shape[0]), (255, 0, 0), 1)

    anchor_y = int(debug_info.get('anchor_cy', 0))
    row_h = debug_info.get('row_unit', 90)
    for i in range(7):
        y = int(anchor_y + (i * row_h))
        cv2.line(vis, (0, y), (img.shape[1], y), (0, 255, 0), 1)
        cv2.putText(vis, f"R{i}", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for s in slots:
        x, y, w, h = s['bbox']
        color = (0, 0, 255) if s['col'] == 0 else (0, 255, 255) 
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        
        # DYNAMICALLY DRAW TEMPLATE ZONES (Based on Config)
        # Colors: Just cycling through basic colors for distinctness
        colors = [(0, 255, 0), (255, 200, 0), (0, 0, 255)] # Green, Cyan, Red
        
        for idx, (r_name, (y1, y2, x1, x2)) in enumerate(TEMPLATE_REGIONS.items()):
            # Scale coordinates from Reference Size (90) to Actual Slot Size (w)
            draw_x1 = int(x + (x1 / ref_w * w))
            draw_y1 = int(y + (y1 / ref_h * w)) # Assuming square aspect ratio for zones
            draw_x2 = int(x + (x2 / ref_w * w))
            draw_y2 = int(y + (y2 / ref_h * w))
            
            c = colors[idx % len(colors)]
            cv2.rectangle(vis, (draw_x1, draw_y1), (draw_x2, draw_y2), c, 1)

        # Labels
        label_text = f"C{s.get('col_cluster','?')}:R{s['row']}"
        if slot_labels and tuple(s['bbox']) in slot_labels:
            clean_name = slot_labels[tuple(s['bbox'])]
            label_text += f" {clean_name}"
        cv2.putText(vis, label_text, (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if save:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"DEBUG_{filename}"), vis)

# ==========================================
#              6. MAIN EXECUTION
# ==========================================

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Hades II Icon Detector")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode (save crops and visual grids)")
    args = parser.parse_args()
    
    # Override Global DEBUG_MODE based on CLI flag
    global DEBUG_MODE
    DEBUG_MODE = args.debug

    # --- STARTUP ---
    if not os.path.exists(ASSETS_DIR):
        print(f"❌ Critical Error: '{ASSETS_DIR}' directory not found.")
        return

    ensure_variants_exist()
    setup_debug_folder()
    
    grid_model, text_reader, libraries, master_lib, mapping = load_resources()
    files = sorted([f for f in glob.glob(os.path.join(INPUT_DIR, "*")) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    
    if not files:
        print(f"❌ No images found in {INPUT_DIR}")
        return

    all_data = []

    print(f"🚀 Processing {len(files)} files (Debug Mode: {DEBUG_MODE})...")

    for i, f in enumerate(files):
        filename = os.path.basename(f)
        img = cv2.imread(f)
        if img is None: continue

        results = grid_model(f, verbose=False)
        raw_slots = [
            (int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]-b.xyxy[0][0]), int(b.xyxy[0][3]-b.xyxy[0][1])) 
            for b in results[0].boxes if b.conf[0] > THRESH_YOLO_CONF
        ]
        
        structured_slots, debug_info = analyze_grid_structure(raw_slots, img.shape[0])
        stats = get_stats(text_reader, img)
        
        row_data = { "Filename": filename, "Clear Time": stats['Clear Time'], "Fear": stats['Fear'] }
        boons_disp = []
        other_boons_raw = []
        debug_crops = []
        slot_labels = {}

        for slot in structured_slots:
            x,y,w,h = slot['bbox']
            crop = img[y:y+h, x:x+w]
            matched = "unknown"
            label = ""

            if slot['col'] == 0:
                cat = COL0_STRUCTURE.get(slot['row'])
                if cat:
                    label = f"C0_R{slot['row']}"
                    matched = match_icon_specific(crop, libraries.get(cat), master_lib, mapping, label)
                    row_data[COL0_DEFAULTS[cat]] = matched
            else:
                label = f"OV_C{slot.get('cc')}"
                matched = match_icon_specific(crop, libraries["overflow_merged"], master_lib, mapping, label)
                if matched != "unknown":
                    other_boons_raw.append(matched)
                    boons_disp.append(f"[C{slot.get('cc')}:R{slot['row']}] {get_display_name(matched, mapping)}")

            display_name = get_display_name(matched, mapping)
            debug_crops.append((crop.copy(), filename, label, display_name))
            slot_labels[(x, y, w, h)] = display_name

        for _, default_val in COL0_DEFAULTS.items():
            if default_val not in row_data:
                row_data[default_val] = default_val
        
        row_data["Other Boons"] = " | ".join(other_boons_raw)
        all_data.append(row_data)

        if DEBUG_MODE:
            for crop_img, base_filename, lab, d_name, in debug_crops:
                clean_label = RE_CLEAN_DEBUG.sub('_', f"{lab}_{d_name}")
                cv2.imwrite(os.path.join(DEBUG_DIR, "crops", f"{base_filename}_{clean_label}.png"), crop_img)
            visualize_grid_logic(img, structured_slots, debug_info, filename, slot_labels=slot_labels, save=True)

        print(f"[{i+1}] {filename} | ⏱️ {stats['Clear Time']} | 💀 {stats['Fear']}")
        print(f"   Core:  {[get_display_name(row_data.get(COL0_DEFAULTS[COL0_STRUCTURE[r]], 'unknown'), mapping) for r in range(7)]}")
        print(f"   Other: {boons_disp}\n" + "-"*60)

    pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Processing complete. CSV saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    main()