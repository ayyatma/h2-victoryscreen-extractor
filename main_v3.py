import cv2
import numpy as np
import glob
import os
import shutil
import pandas as pd
import easyocr
import re
from ultralytics import YOLO

# --- CONFIG ---
INPUT_FOLDER = "vic_screens"
REFERENCE_BASE = "reference_icons"
MODEL_PATH = "runs/detect/train/weights/best.pt"
OUTPUT_CSV = "hades_run_data.csv"

# OVERLAYS
PIN_OVERLAY_PATH = "pin_overlay.png"
RANK_FILES = ["rank_3.png"]

# --- DEBUG SETTINGS ---
DEBUG_MODE = True
DEBUG_FOLDER = "debug_output"

# --- 1. VISUALIZATION (UPDATED) ---
def setup_debug_folder():
    if DEBUG_MODE:
        if os.path.exists(DEBUG_FOLDER): shutil.rmtree(DEBUG_FOLDER)
        os.makedirs(DEBUG_FOLDER)
        os.makedirs(os.path.join(DEBUG_FOLDER, "crops"))

def visualize_grid_logic(img, slots, debug_info, filename, slot_labels=None, save=True):
    vis = img.copy()
    
    # 1. Grid Lines
    for col_x in debug_info.get('col_centers', []):
        cv2.line(vis, (int(col_x), 0), (int(col_x), img.shape[0]), (255, 0, 0), 1)

    anchor_y = int(debug_info.get('anchor_cy', 0))
    row_h = debug_info.get('row_unit', 90)
    for i in range(7):
        y = int(anchor_y + (i * row_h))
        cv2.line(vis, (0, y), (img.shape[1], y), (0, 255, 0), 1)
        cv2.putText(vis, f"R{i}", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 2. Draw Slots & Zones
    for s in slots:
        x, y, w, h = s['bbox']
        color = (0, 0, 255) if s['col'] == 0 else (0, 255, 255) 
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        
        icon_size = w 
        
        # 1. HEAD (Green) - User Defined Safe Zone
        h_x1, h_y1 = int(x + (30/90*w)), int(y + (15/90*w))
        h_x2, h_y2 = int(x + (75/90*w)), int(y + (57/90*w))
        cv2.rectangle(vis, (h_x1, h_y1), (h_x2, h_y2), (0, 255, 0), 1)
        
        # 2. HEART (Blue) - User Defined Safe Zone
        b_x1, b_y1 = int(x + (20/90*w)), int(y + (25/90*w))
        b_x2, b_y2 = int(x + (80/90*w)), int(y + (57/90*w))
        cv2.rectangle(vis, (b_x1, b_y1), (b_x2, b_y2), (255, 200, 0), 1)

        # 3. BROAD (Red) - The "Big" Region
        # Logic: y=15:75, x=15:75
        br_x1, br_y1 = int(x + (15/90*w)), int(y + (15/90*w))
        br_x2, br_y2 = int(x + (75/90*w)), int(y + (75/90*w))
        cv2.rectangle(vis, (br_x1, br_y1), (br_x2, br_y2), (0, 0, 255), 1)

        # Put basic slot coord
        cv2.putText(vis, f"C{s.get('col_cluster','?')}:R{s['row']}", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # If a label for this slot was provided, draw it
        if slot_labels:
            key = (x, y, w, h)
            lbl = slot_labels.get(key) or slot_labels.get(str(key)) or slot_labels.get(s.get('row'))
            if lbl:
                txt = str(lbl)
                cv2.putText(vis, txt, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if save and DEBUG_MODE:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"DEBUG_{filename}"), vis)
    return vis

def save_debug_crop(crop, filename, label):
    if not DEBUG_MODE: return
    clean_label = re.sub(r'[^\w\-_\. ]', '_', label)
    path = os.path.join(DEBUG_FOLDER, "crops", f"{filename}_{clean_label}.png")
    cv2.imwrite(path, crop)

# --- 2. VARIANTS ---
def overlay_image_alpha(img, overlay, opacity=1.0):
    h, w = img.shape[:2]
    if overlay.shape[:2] != (h, w): overlay = cv2.resize(overlay, (w, h))
    if img.shape[2] == 4:
        img_bgr, img_alpha = img[:, :, :3], img[:, :, 3] / 255.0
    else:
        img_bgr, img_alpha = img, np.ones((h, w))
    
    if overlay.shape[2] == 4:
        ov_bgr, ov_alpha = overlay[:, :, :3], (overlay[:, :, 3] / 255.0) * opacity
    else: return img 

    out_alpha = ov_alpha + img_alpha * (1 - ov_alpha)
    safe_alpha = np.maximum(out_alpha, 1e-6)
    out_bgr = (ov_bgr * ov_alpha[:, :, None] + img_bgr * img_alpha[:, :, None] * (1 - ov_alpha)[:, :, None]) / safe_alpha[:, :, None]
    return cv2.merge([np.clip(out_bgr,0,255).astype(np.uint8)[:,:,i] for i in range(3)] + [np.clip(out_alpha*255,0,255).astype(np.uint8)])

def ensure_variants_exist():
    print("🔨 Checking/Generating Variants...")
    pin_overlay = cv2.imread(PIN_OVERLAY_PATH, cv2.IMREAD_UNCHANGED) if os.path.exists(PIN_OVERLAY_PATH) else None
    rank_overlays = {}
    for r_file in RANK_FILES:
        if os.path.exists(r_file): rank_overlays[r_file.replace("rank_", "").replace(".png", "")] = cv2.imread(r_file, cv2.IMREAD_UNCHANGED)
    
    folders = [os.path.join(REFERENCE_BASE, cat) for cat in COL0_STRUCTURE.values()] + [os.path.join(REFERENCE_BASE, "boons"), os.path.join(REFERENCE_BASE, "keepsakes")]

    count = 0
    for folder in folders:
        # if not os.path.exists(folder): continue
        is_keepsake = "keepsakes" in os.path.basename(folder).lower()
        files = glob.glob(os.path.join(folder, "*"))
        base_files = [f for f in files if "_PINNED" not in f and "_RANK" not in f and f.lower().endswith(('.png', '.jpg'))]

        for f in base_files:
            name, path = os.path.splitext(os.path.basename(f))[0], os.path.dirname(f)
            # if os.path.exists(os.path.join(path, f"{name}_PINNED.png")): continue 
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None: continue
            
            if pin_overlay is not None:
                cv2.imwrite(os.path.join(path, f"{name}_PINNED.png"), overlay_image_alpha(img, pin_overlay, 0.9))
                count += 1
            if is_keepsake and rank_overlays:
                for k, v in rank_overlays.items():
                    ranked = overlay_image_alpha(img, v, 1.0)
                    cv2.imwrite(os.path.join(path, f"{name}_RANK_{k}.png"), ranked)
                    if pin_overlay is not None:
                        cv2.imwrite(os.path.join(path, f"{name}_PINNED_RANK_{k}.png"), overlay_image_alpha(ranked, pin_overlay, 0.9))
                    count += 2
    if count: print(f"✨ Generated {count} variants.")

# --- 3. GRID LOGIC ---
def analyze_grid_structure(slots, img_w, img_h):
    if not slots: return [], {}
    avg_w, avg_h = np.mean([s[2] for s in slots]), np.mean([s[3] for s in slots])
    slots_wc = [{'bbox': s, 'cx': s[0]+s[2]/2, 'cy': s[1]+s[3]/2} for s in slots]
    slots_wc.sort(key=lambda s: s['cx'])
    
    cols, cur_col = [], [slots_wc[0]]
    for i in range(1, len(slots_wc)):
        if (slots_wc[i]['cx'] - slots_wc[i-1]['cx']) > (avg_w * 0.8):
            cols.append(cur_col)
            cur_col = [slots_wc[i]]
        else: cur_col.append(slots_wc[i])
    cols.append(cur_col)

    gaps = []
    for c in cols:
        if len(c) < 2: continue
        c.sort(key=lambda s: s['cy'])
        for k in range(len(c)-1):
            dist = c[k+1]['cy'] - c[k]['cy']
            if (avg_h * 0.9) < dist < (avg_h * 1.4): gaps.append(dist)
    
    row_unit = np.median(gaps) if gaps else avg_h * 1.15
    if not cols: return [], {}
    
    cols[0].sort(key=lambda s: s['cy'])
    anchor_y = cols[0][0]['cy']
    
    final, col0_occ, col_centers = [], {}, []
    
    for idx, c in enumerate(cols):
        col_centers.append(np.mean([item['cx'] for item in c]))
        for item in c:
            r = int(round((item['cy'] - anchor_y) / row_unit))
            if idx == 0:
                if 0 <= r <= 6:
                    score = abs((item['cy'] - anchor_y) - (r * row_unit))
                    new = {'bbox': item['bbox'], 'col': 0, 'row': r, 'score': score, 'col_cluster': 0}
                    if r in col0_occ:
                        if new['score'] < col0_occ[r]['score']:
                            old = col0_occ[r]
                            old['col'] = 1
                            final.append(old)
                            col0_occ[r] = new
                        else:
                            new['col'] = 1
                            final.append(new)
                    else: col0_occ[r] = new
                else: final.append({'bbox': item['bbox'], 'col': 1, 'row': r, 'col_cluster': 0})
            else: final.append({'bbox': item['bbox'], 'col': 1, 'row': r, 'col_cluster': idx})
            
    final.extend(col0_occ.values())
    final.sort(key=lambda s: (s['col'], s['row'], s.get('col_cluster', 0))) 
    return final, {'col_centers': col_centers, 'row_unit': row_unit, 'anchor_cy': anchor_y}

# --- 4. RESOURCES & MATCHING ---
COL0_STRUCTURE = { 0: "aspects", 1: "familiars", 2: "attacks", 3: "specials", 4: "casts", 5: "sprints", 6: "gains" }
COL0_DEFAULTS = { "aspects": "Aspect", "familiars": "Familiar", "attacks": "Attack", "specials": "Special", "casts": "Cast", "sprints": "Sprint", "gains": "Magick" }

def load_display_map(path="iconname_code_display.csv"):
    d = {}
    if not os.path.exists(path):
        return d
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = str(row.get('icon_filename', '')).strip()
            val = str(row.get('display_text', '')).strip()
            if not key:
                continue
            # register multiple key variants: with/without extension and cleaned forms
            d[key] = val
            base = os.path.splitext(key)[0]
            d[base] = val
            cleaned = re.sub(r'_PINNED_RANK_\d+', '', base)
            cleaned = re.sub(r'_RANK_\d+', '', cleaned)
            cleaned = cleaned.replace("_PINNED", "")
            d[cleaned] = val
            d[cleaned + '.png'] = val
    except Exception as e:
        print(f"⚠️ Failed to load display map '{path}': {e}")
    return d

def get_display_name(iconname, display_map):
    if not iconname or iconname == "unknown":
        return iconname
    # try direct matches first
    if iconname in display_map:
        return display_map[iconname]
    base = clean_string(iconname)
    if base in display_map:
        return display_map[base]
    if (base + '.png') in display_map:
        return display_map[base + '.png']
    if (iconname + '.png') in display_map:
        return display_map[iconname + '.png']
    return base

def detect_environment(img):
    h, w = img.shape[:2]
    banner = img[:int(h*0.15), :]
    return "Underworld" if np.mean(banner[:,:,1]) > (np.mean(banner[:,:,2]) + 30) else "Surface"

def load_resources():
    ensure_variants_exist()
    print("⏳ Loading AI Models...")
    if not os.path.exists(MODEL_PATH): return None, None, None, None, {}
    grid_model = YOLO(MODEL_PATH)
    text_reader = easyocr.Reader(['en'], gpu=True)
    libs, master = {}, {}
    for k, v in COL0_STRUCTURE.items():
        libs[v] = load_icon_folder(os.path.join(REFERENCE_BASE, v))
        master.update(libs[v])
    libs["boons"] = load_icon_folder(os.path.join(REFERENCE_BASE, "boons"))
    master.update(libs["boons"])
    libs["keepsakes"] = load_icon_folder(os.path.join(REFERENCE_BASE, "keepsakes"))
    master.update(libs["keepsakes"])
    print(f"✅ Loaded {len(master)} icons.")
    display_map = load_display_map()
    if display_map: print(f"🔎 Loaded {len(display_map)} display mappings.")
    return grid_model, text_reader, libs, master, display_map

def load_icon_folder(path):
    d = {}
    if not os.path.exists(path): return d
    for f in glob.glob(os.path.join(path, "*")):
        if f.lower().endswith(('.png','.jpg','.jpeg')):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None: d[os.path.splitext(os.path.basename(f))[0]] = cv2.resize(img, (90, 90))
    return d

def match_icon_specific(crop_img, library, master_lib=None, debug_label="Item"):
    if not library: return "unknown", 0.0
    scene = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    h, w = scene.shape
    if h > (w * 1.15): scene = scene[:w, :] 
    
    scale = 100 / scene.shape[1]
    scene = cv2.resize(scene, (100, int(scene.shape[0] * scale)))
    
    candidates = [] 
    
    for name, ref_full in library.items():
        # --- MULTI-ZONE TEMPLATES ---
        
        # 1. HEAD (Strict/Safe): User defined.
        t_head = ref_full[15:57, 30:75]
        
        # 2. HEART (Strict/Safe): User defined.
        t_heart = ref_full[25:57, 20:80]
        
        # 3. BROAD (Risky/Full): 
        # y=15:80 (Top to Bottom Text)
        # x=15:85 (Full Width, slight shave for Pin)
        # Catches the "Whole Picture". 
        # If text is present, this scores low (and is ignored).
        # If text is absent, this scores high (and confirms match).
        t_broad = ref_full[15:75, 15:75]
        
        local_best = 0.0
        for tmpl in [t_head, t_heart, t_broad]:
            for s in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
                th, tw = tmpl.shape
                cw, ch = int(tw*s), int(th*s)
                if cw >= scene.shape[1] or ch >= scene.shape[0]: continue
                
                res = cv2.matchTemplate(scene, cv2.resize(tmpl, (cw, ch)), cv2.TM_CCOEFF_NORMED)
                score = np.max(res)
                if score > local_best: local_best = score
        
        if local_best > 0.4: candidates.append((local_best, name))

    if not candidates: return "unknown"
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name = candidates[0]
    
    # Smart Gap
    def get_base_name(s):
        s = re.sub(r'_PINNED_RANK_\d+', '', s)
        s = re.sub(r'_RANK_\d+', '', s)
        return s.replace("_PINNED", "")

    base_winner = get_base_name(best_name)
    second_score = 0.0
    for score, name in candidates[1:]:
        if get_base_name(name) != base_winner:
            second_score = score
            break
    real_gap = best_score - second_score


    top_5 = candidates[:5]
    print(f"⚠️  [DEBUG] '{debug_label}' Unmatched (Best: {best_score:.3f}, Gap: {real_gap:.3f}). Candidates:")
    for score, name in top_5: print(f"      - {name}: {score:.3f}")

    # Thresholds
    if best_score > 0.70: return best_name
    if best_score > 0.65 and real_gap > 0.05: return best_name
    

    
    if master_lib: return match_icon_specific(crop_img, master_lib, None, debug_label + " (Fallback)")
    return "unknown"

def clean_string(s):
    s = re.sub(r'_PINNED_RANK_\d+', '', s)
    s = re.sub(r'_RANK_\d+', '', s)
    return s.replace("_PINNED", "")

# --- 5. STATS ---
def get_stats_ai(reader, img):
    h, w, _ = img.shape
    stats = {'Clear Time': 'Unknown', 'Fear': '0'}
    crop = img[int(h*0.15):int(h*0.35), int(w*0.75):w]
    res = reader.readtext(crop, detail=0, allowlist='0123456789:.,UsedFar ')
    time_match = re.search(r'(\d{1,2})[:.,](\d{2})', "".join(res))
    if time_match: stats['Clear Time'] = f"{time_match.group(1)}:{time_match.group(2)}"
    for t in res:
        fm = re.search(r'Used.*?(\d+)', t)
        if fm: stats['Fear'] = fm.group(1)
        elif t.isdigit() and len(t)<4 and stats['Fear']=='0': stats['Fear'] = t
    return stats

# --- 6. MAIN ---
def main():
    setup_debug_folder()
    grid_model, text_reader, libraries, master_lib, display_map = load_resources()
    if not grid_model: return
    files = sorted([f for f in glob.glob(os.path.join(INPUT_FOLDER, "*")) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    
    all_data = []
    print(f"\n🚀 PROCESSING {len(files)} FILES...\n")

    for i, f in enumerate(files):
        filename = os.path.basename(f)
        img = cv2.imread(f)
        if img is None: continue
        
        results = grid_model(f, verbose=False)
        raw = []
        for r in results:
            for b in r.boxes:
                if b.conf[0] > 0.4:
                    x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                    raw.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        
        slots, debug = analyze_grid_structure(raw, img.shape[1], img.shape[0])
        # Defer writing debug visual and crops until after labeling
        debug_crops = []
        slot_labels = {}
        stats = get_stats_ai(text_reader, img)
        env = detect_environment(img)
        
        row_data = { "Filename": filename, "Region": env, "Clear Time": stats['Clear Time'], "Fear": stats['Fear'] }
        boons_console = []
        other_boons = []
        col0_logs = {} 
        
        for slot in slots:
            x,y,w,h = slot['bbox']
            crop = img[y:y+h, x:x+w]
            
            matched = "unknown"
            label = ""
            
            if slot['col'] == 0:
                cat = COL0_STRUCTURE.get(slot['row'])
                label = f"C0_R{slot['row']}_{cat}"
                if cat:
                    matched = match_icon_specific(crop, libraries.get(cat), None, label)
                    
                    if matched == "unknown":
                        fallback = match_icon_specific(crop, master_lib, None, label + "_CheckMaster")
                        if fallback != "unknown":
                            display_fb = get_display_name(fallback, display_map)
                            col0_logs[slot['row']] = f"⚠️ WRONG FOLDER: '{display_fb}'"
                            row_data[COL0_DEFAULTS[cat]] = "UNKNOWN"
                        else:
                            col0_logs[slot['row']] = "[UNKNOWN]" 
                            row_data[COL0_DEFAULTS[cat]] = "UNKNOWN"
                    else:
                        clean = clean_string(matched)
                        display = get_display_name(matched, display_map)
                        col0_logs[slot['row']] = display
                        row_data[COL0_DEFAULTS[cat]] = clean
            else:
                label = f"OV_C{slot.get('col_cluster','?')}_R{slot['row']}"
                matched = match_icon_specific(crop, libraries["boons"], master_lib, label)
                if matched == "unknown": matched = match_icon_specific(crop, libraries["keepsakes"], master_lib, label)
                
                if matched != "unknown":
                    clean = clean_string(matched)
                    other_boons.append(clean)
                    display = get_display_name(matched, display_map)
                    boons_console.append(f"[C{slot.get('col_cluster','?')}:R{slot['row']}] {display}")
                else:
                    boons_console.append(f"[C{slot.get('col_cluster','?')}:R{slot['row']}] UNKNOWN")

            save_name = matched if matched != "unknown" else "UNKNOWN"
            debug_name = get_display_name(save_name, display_map) if save_name != "UNKNOWN" else "UNKNOWN"
            # collect crop and intended debug filename (use bbox as key for labeling overlay)
            debug_crops.append((crop.copy(), filename, label, debug_name, (x, y, w, h)))
            slot_labels[(x, y, w, h)] = (get_display_name(matched, display_map) if matched != "unknown" else "UNKNOWN")

        for _, v in COL0_DEFAULTS.items():
            if v not in row_data: row_data[v] = "EMPTY"

        row_data["Other Boons"] = " | ".join(other_boons)
        all_data.append(row_data)

        # Now write collected crops using display names
        if DEBUG_MODE:
            for crop_img, base_filename, lab, debug_name, bbox in debug_crops:
                clean_label = re.sub(r"[^\w\-\_\. ]", '_', f"{lab}_{debug_name}")
                out_path = os.path.join(DEBUG_FOLDER, "crops", f"{base_filename}_{clean_label}.png")
                cv2.imwrite(out_path, crop_img)

        # Render and save debug visual with labels now that we have slot labels
        visualize_grid_logic(img, slots, debug, filename, slot_labels=slot_labels, save=True)

        print(f"[{i+1}] {filename}")
        print(f"   Core:  {[col0_logs.get(r, '---') for r in range(7)]}")
        print(f"   Other: {boons_console}")
        print("-" * 60)

    pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved to {OUTPUT_CSV}. Check /{DEBUG_FOLDER}")

if __name__ == "__main__":
    main()