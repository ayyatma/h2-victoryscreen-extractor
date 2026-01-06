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
from difflib import SequenceMatcher

# ==========================================
#        1. CENTRAL CONFIGURATION
# ==========================================

# --- DIRECTORIES ---
ASSETS_DIR = "assets"
INPUT_DIR = "vic_screens"
DEBUG_DIR = "debug"

# --- FILE PATHS ---
MODEL_FILE = "best.pt"
MAPPING_FILE = "iconname_code_display.csv"
PIN_OVERLAY_FILE = "pin_overlay.png"
RANK_FILES = ["rank_1.png", "rank_2.png", "rank_3.png", "rank_4.png"]
REF_ICONS_SUBDIR = "reference_icons"
OUTPUT_CSV = "hades_run_data.csv"

# --- DEBUGGING DEFAULT ---
DEBUG_MODE = False

# --- IMAGE PROCESSING ---
TARGET_ICON_SIZE = (90, 90)
MATCH_SCENE_WIDTH = int(TARGET_ICON_SIZE[0] * 1.15) 
MATCH_SCALES = [0.7, 0.8, 0.9, 1.0, 1.1]
ICON_ASPECT_RATIO_LIMIT = 1.15

# --- EXCLUSION LIST ---
NO_PIN_GENERATION = [] 

# --- OCR CONFIGURATION ---
OCR_LANGUAGES = ['en'] 
# NOTE: Fuzzy matching now handles these concepts, lists are less critical but kept for structure
OCR_SEARCH_REGION = (0.0, 0.40, 0.60, 1.0) 

# --- ENVIRONMENT DETECTION CONFIG ---
ENV_BANNER_RATIO = 0.15      
ENV_COLOR_OFFSET = 30        

# --- GRID ANALYSIS CONFIG ---
GRID_COL_CLUSTER_THRESH = 0.8 
GRID_ROW_GAP_MIN = 0.9        
GRID_ROW_GAP_MAX = 1.4        
GRID_ROW_UNIT_DEFAULT = 1.15  

# --- MATCHING REGIONS ---
TEMPLATE_REGIONS = {
    "head":  (15, 57, 30, 75), 
    "heart": (25, 57, 20, 80), 
    "broad": (15, 80, 15, 85)
}

# --- THRESHOLDS ---
THRESH_YOLO_CONF = 0.4
THRESH_TEMPLATE_MIN = 0.4
THRESH_SCORE_HIGH = 0.70
THRESH_SCORE_MED = 0.65
THRESH_GAP_MIN = 0.025

# --- LOGIC CONSTANTS ---
COL0_STRUCTURE = { 0: "aspects", 1: "familiars", 2: "attacks", 3: "specials", 4: "casts", 5: "sprints", 6: "gains" }
COL0_DEFAULTS = { "aspects": "Aspect", "familiars": "Familiar", "attacks": "Attack", "specials": "Special", "casts": "Cast", "sprints": "Sprint", "gains": "Magick" }

# --- REGEX ---
RE_PINNED = re.compile(r'_PINNED')
RE_RANK = re.compile(r'_RANK_\d+')
RE_CLEAN_DEBUG = re.compile(r'[^\w\-\_\. ]')


# ==========================================
#           2. HELPER FUNCTIONS
# ==========================================

def get_asset_path(filename):
    return os.path.join(ASSETS_DIR, filename)

def clean_name_internal(internal_name):
    name = RE_PINNED.sub('', internal_name)
    name = RE_RANK.sub('', name)
    return name

def get_display_name(internal_name, mapping):
    if internal_name == "unknown": return "UNKNOWN"
    base_key = clean_name_internal(internal_name)
    return mapping.get(base_key, base_key)

def overlay_image_alpha(img, overlay, opacity=1.0):
    h, w = img.shape[:2]
    if overlay.shape[:2] != (h, w): overlay = cv2.resize(overlay, (w, h))
        
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
    
    return cv2.merge([np.clip(out_bgr, 0, 255).astype(np.uint8)[:,:,i] for i in range(3)] + [np.clip(out_alpha * 255, 0, 255).astype(np.uint8)])


# ==========================================
#           3. RESOURCE LOADING
# ==========================================

def ensure_variants_exist():
    print("🔨 Checking/Generating Variants...")
    pin_path = get_asset_path(PIN_OVERLAY_FILE)
    if not os.path.exists(pin_path): 
        print(f"⚠️ Pin overlay missing at {pin_path}. Skipping.")
        return
    
    pin_overlay = cv2.imread(pin_path, cv2.IMREAD_UNCHANGED)
    rank_overlays = {}
    for r_file in RANK_FILES:
        path = get_asset_path(r_file)
        if os.path.exists(path):
            key = r_file.replace("rank_", "").replace(".png", "")
            rank_overlays[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    ref_base = get_asset_path(REF_ICONS_SUBDIR)
    # We only generate PINNED variants for Boons/Keepsakes and Core slots, not Aspects/Familiars usually.
    # Adjust categories if needed based on NO_PIN_GENERATION list.
    categories = ["attacks", "specials", "casts", "sprints", "gains", "boons", "keepsakes"]
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
            
            if name in NO_PIN_GENERATION: continue
            
            if os.path.exists(os.path.join(path, f"{name}_PINNED.png")): continue 
            
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None: continue
            
            cv2.imwrite(os.path.join(path, f"{name}_PINNED.png"), overlay_image_alpha(img, pin_overlay, 1))
            count += 1
            
            if is_keepsake and rank_overlays:
                for r_key, r_img in rank_overlays.items():
                    ranked = overlay_image_alpha(img, r_img, 1.0)
                    cv2.imwrite(os.path.join(path, f"{name}_RANK_{r_key}.png"), ranked)
                    cv2.imwrite(os.path.join(path, f"{name}_PINNED_RANK_{r_key}.png"), overlay_image_alpha(ranked, pin_overlay, 1))
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

def load_resources(use_gpu):
    print(f"⏳ Loading AI Models (GPU: {use_gpu})...")
    
    if not os.path.exists(ASSETS_DIR):
        print(f"❌ ERROR: Assets directory '{ASSETS_DIR}' not found!")
        exit()

    model_path = get_asset_path(MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        exit()
    
    device_str = 'cuda:0' if use_gpu else 'cpu'
    grid_model = YOLO(model_path)
    grid_model.to(device_str)
    
    text_reader = easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu)
    
    mapping = {}
    mapping_path = get_asset_path(MAPPING_FILE)
    if os.path.exists(mapping_path):
        df = pd.read_csv(mapping_path)
        mapping = {str(row['icon_filename']).replace('.png', ''): str(row['display_text']) 
                   for _, row in df.iterrows()}
    else:
        print(f"⚠️ Mapping file not found at {mapping_path}.")

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
    
    libs["overflow_merged"] = {**libs["boons"], **libs["keepsakes"]}
    
    print(f"✅ Loaded {len(master)} icons.")
    return grid_model, text_reader, libs, master, mapping


# ==========================================
#           4. CORE LOGIC
# ==========================================

def detect_environment(img):
    h, w = img.shape[:2]
    banner = img[:int(h * ENV_BANNER_RATIO), :]
    if np.mean(banner[:,:,1]) > (np.mean(banner[:,:,2]) + ENV_COLOR_OFFSET):
        return "Underworld"
    return "Surface"

# --- UPDATED OCR FUNCTION WITH BETTER DEBUG POSITIONING ---
def get_stats_dynamic(reader, img):
    h, w = img.shape[:2]
    stats = {'Clear Time': 'Unknown', 'Fear': '0'}
    debug_boxes = []
    
    # Track the leftmost anchor point found to position debug text
    anchor_left_x = w 
    anchor_top_y = 0

    # 1. Define Search Region (Top-Right)
    y_start, y_end, x_start, x_end = OCR_SEARCH_REGION
    y1, y2 = int(h * y_start), int(h * y_end)
    x1, x2 = int(w * x_start), int(w * x_end)
    
    broad_crop = img[y1:y2, x1:x2]
    
    # 2. Pre-process: Upscale & Smart Binarization (Otsu)
    gray_crop = cv2.cvtColor(broad_crop, cv2.COLOR_BGR2GRAY)
    
    # Upscale 2x -> 3x (Better for very small text like "Used")
    gray_crop = cv2.resize(gray_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
    
    # --- FIX: Gentler Blur ---
    # Was (5,5), which erased thin lines. Now (3,3) just to remove noise.
    gray_blur = cv2.GaussianBlur(gray_crop, (3,3), 0)
    
    _, bin_crop = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Read EVERYTHING
    results = reader.readtext(bin_crop)
    
    if DEBUG_MODE:
        cv2.imwrite(os.path.join(DEBUG_DIR, "debug_ocr_otsu.png"), bin_crop)
        print("--- OCR RAW DUMP ---")
        for (_, text, _) in results: print(f"Read: '{text}'")
        print("--------------------")

    def get_cy(bbox): return (bbox[0][1] + bbox[2][1]) / 2

    # --- FUZZY MATCHING HELPER ---
    def is_fuzzy_match(text_found, target, threshold=0.6):
        clean_found = re.sub(r'[^A-Za-z]', '', text_found).lower()
        clean_target = target.lower()
        similarity = SequenceMatcher(None, clean_found, clean_target).ratio()
        
        # Substring Check (covers "ResultsClearTime" merging)
        is_substring = False
        if len(clean_found) > len(clean_target) + 3:
            is_substring = clean_target in clean_found
            
        is_match = (similarity >= threshold) or is_substring
        
        # Lower debug print threshold to 0.3 so you can see "almost" matches
        if DEBUG_MODE and (is_match or similarity > 0.3):
            match_str = "✅ MATCH" if is_match else "❌ NO"
            print(f"   🔍 [Fuzzy] '{text_found}' vs '{target}' = {similarity:.3f} ({match_str})")
        return is_match

    # 4. Spatial Parsing
    for i, (bbox_anchor, text_anchor, _) in enumerate(results):
        anchor_cy = get_cy(bbox_anchor)
        found_anchor = False
        
        # --- CASE 1: CLEAR TIME ---
        # Threshold 0.60 allows for 1-2 wrong letters in "ClearTime"
        if is_fuzzy_match(text_anchor, "ClearTime", threshold=0.60) or \
           is_fuzzy_match(text_anchor, "Clear", threshold=0.8):
            found_anchor = True
            for j in range(len(results)):
                if i == j: continue
                bbox_val, text_val, _ = results[j]
                
                # Check Right (with scaled offset since we upscaled 3x)
                if bbox_val[0][0] < (bbox_anchor[1][0] - 10): continue
                if abs(get_cy(bbox_val) - anchor_cy) < 25:
                    
                    if re.search(r'\d{1,2}[:.,]\d{2}', text_val):
                        clean_time = text_val.replace(',', '.').replace(':', '.')
                        parts = clean_time.split('.')
                        
                        if len(parts) >= 3:
                            stats['Clear Time'] = f"{parts[0]}:{parts[1]}.{parts[2]}"
                        elif len(parts) == 2:
                            stats['Clear Time'] = f"{parts[0]}:{parts[1]}"
                        else:
                            stats['Clear Time'] = text_val
                        
                        # Scale back down by 3.0
                        bx1, by1 = int(bbox_val[0][0]/3)+x1, int(bbox_val[0][1]/3)+y1
                        bx2, by2 = int(bbox_val[2][0]/3)+x1, int(bbox_val[2][1]/3)+y1
                        debug_boxes.append((bx1, by1, bx2, by2))
                        break

        # --- CASE 2: FEAR / USED ---
        # "Used" is short, so we need a stricter threshold (0.7) to avoid false positives on 4-letter noise
        elif is_fuzzy_match(text_anchor, "Used", threshold=0.7):
            found_anchor = True
            for j in range(len(results)):
                if i == j: continue
                bbox_val, text_val, _ = results[j]
                
                if bbox_val[0][0] < (bbox_anchor[1][0] - 10): continue
                if abs(get_cy(bbox_val) - anchor_cy) < 25:
                    
                    digits = re.sub(r'\D', '', text_val)
                    if digits.isdigit() and len(digits) > 0:
                        stats['Fear'] = digits
                        
                        bx1, by1 = int(bbox_val[0][0]/3)+x1, int(bbox_val[0][1]/3)+y1
                        bx2, by2 = int(bbox_val[2][0]/3)+x1, int(bbox_val[2][1]/3)+y1
                        debug_boxes.append((bx1, by1, bx2, by2))
                        break
        
        if found_anchor:
            ax1 = int(bbox_anchor[0][0]/3) + x1
            ay1 = int(bbox_anchor[0][1]/3) + y1
            if ax1 < anchor_left_x:
                anchor_left_x = ax1
                anchor_top_y = ay1

    return stats, debug_boxes, (anchor_left_x, anchor_top_y)


def match_icon_specific(crop_img, library, master_lib, mapping, debug_label):
    if not library: return "unknown"
    
    scene = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    if scene.shape[0] > (scene.shape[1] * ICON_ASPECT_RATIO_LIMIT): 
        scene = scene[:scene.shape[1], :] 
    
    scale_factor = MATCH_SCENE_WIDTH / scene.shape[1]
    scene = cv2.resize(scene, (MATCH_SCENE_WIDTH, int(scene.shape[0] * scale_factor)))
    
    candidates = [] 
    
    for name, ref in library.items():
        best_tmpl_score = 0.0
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
    
    if DEBUG_MODE:
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
        if (slots_wc[i]['cx'] - slots_wc[i-1]['cx']) > (avg_w * GRID_COL_CLUSTER_THRESH):
            cols.append(cur); cur = [slots_wc[i]]
        else: cur.append(slots_wc[i])
    cols.append(cur)
    
    gaps = []
    for c in cols:
        if len(c) < 2: continue
        c.sort(key=lambda s: s['cy'])
        for k in range(len(c)-1):
            d = c[k+1]['cy'] - c[k]['cy']
            if (avg_h * GRID_ROW_GAP_MIN) < d < (avg_h * GRID_ROW_GAP_MAX): 
                gaps.append(d)
    
    row_unit = np.median(gaps) if gaps else avg_h * GRID_ROW_UNIT_DEFAULT
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


# ==========================================
#         5. VISUALIZATION & DEBUG
# ==========================================

def setup_debug_folder():
    if DEBUG_MODE:
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        os.makedirs(os.path.join(DEBUG_DIR, "crops"))

# --- UPDATED VISUALIZATION TO USE ANCHOR POSITION ---
def visualize_grid_logic(img, slots, debug_info, filename, slot_labels=None, debug_boxes=None, stats_text=None, anchor_pos=None, save=True):
    vis = img.copy()
    ref_w, ref_h = TARGET_ICON_SIZE

    # Draw grid lines
    for col_x in debug_info.get('col_centers', []):
        cv2.line(vis, (int(col_x), 0), (int(col_x), img.shape[0]), (255, 0, 0), 1)

    anchor_y = int(debug_info.get('anchor_cy', 0))
    row_h = debug_info.get('row_unit', 90)
    for i in range(7):
        y = int(anchor_y + (i * row_h))
        cv2.line(vis, (0, y), (img.shape[1], y), (0, 255, 0), 1)
        cv2.putText(vis, f"R{i}", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw slots and labels
    for s in slots:
        x, y, w, h = s['bbox']
        color = (0, 0, 255) if s['col'] == 0 else (0, 255, 255) 
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        
        colors = [(0, 255, 0), (255, 200, 0), (0, 0, 255)]
        for idx, (r_name, (y1, y2, x1, x2)) in enumerate(TEMPLATE_REGIONS.items()):
            draw_x1 = int(x + (x1 / ref_w * w))
            draw_y1 = int(y + (y1 / ref_h * w))
            draw_x2 = int(x + (x2 / ref_w * w))
            draw_y2 = int(y + (y2 / ref_h * w))
            c = colors[idx % len(colors)]
            cv2.rectangle(vis, (draw_x1, draw_y1), (draw_x2, draw_y2), c, 1)

        label_text = f"C{s.get('col_cluster','?')}:R{s['row']}"
        if slot_labels and tuple(s['bbox']) in slot_labels:
            clean_name = slot_labels[tuple(s['bbox'])]
            label_text += f" {clean_name}"
        cv2.putText(vis, label_text, (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # --- Draw OCR Debug Info ---
    if debug_boxes:
        # Draw rectangles around the detected VALUES (e.g. the time numbers)
        for (x1, y1, x2, y2) in debug_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2) 
            
        # Draw the stats text using the ANCHOR position (left side) so it doesn't cut off
        if stats_text and anchor_pos and anchor_pos[0] < img.shape[1]:
            # anchor_pos is (left_x, top_y) of the leftmost anchor found.
            # Position text slightly above it.
            tx, ty = anchor_pos[0], max(30, anchor_pos[1] - 10)
            cv2.putText(vis, stats_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if save:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"DEBUG_{filename}"), vis)


# ==========================================
#              6. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    args = parser.parse_args()
    
    global DEBUG_MODE
    DEBUG_MODE = args.debug

    if not os.path.exists(ASSETS_DIR):
        print(f"❌ Critical Error: '{ASSETS_DIR}' directory not found.")
        return

    ensure_variants_exist()
    setup_debug_folder()
    
    grid_model, text_reader, libraries, master_lib, mapping = load_resources(args.gpu)
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
        
        # --- Get stats AND anchor position ---
        stats, debug_boxes, anchor_pos = get_stats_dynamic(text_reader, img)
        env = detect_environment(img)
        
        row_data = { "Filename": filename, "Region": env, "Clear Time": stats['Clear Time'], "Fear": stats['Fear'] }
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
                    matched = match_icon_specific(crop, libraries.get(cat), None, mapping, label)
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
            
            stats_str = f"Time: {stats['Clear Time']} | Fear: {stats['Fear']}"
            # --- Pass anchor_pos to visualization ---
            visualize_grid_logic(img, structured_slots, debug_info, filename, 
                               slot_labels=slot_labels, debug_boxes=debug_boxes, 
                               stats_text=stats_str, anchor_pos=anchor_pos, save=True)

        print(f"[{i+1}] {filename} ({env}) | ⏱️ {stats['Clear Time']} | 💀 {stats['Fear']}")
        print(f"   Core:  {[get_display_name(row_data.get(COL0_DEFAULTS[COL0_STRUCTURE[r]], 'unknown'), mapping) for r in range(7)]}")
        print(f"   Other: {boons_disp}\n" + "-"*60)

    pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Processing complete. CSV saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    main()