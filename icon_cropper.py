from PIL import Image
import os

# --- SETTINGS ---
TARGET_SIZE = 90  # The final width and height in pixels
INPUT_FOLDER = "icons"
OUTPUT_FOLDER = "icons_resized"
# ----------------

def process_icons():
    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    valid_extensions = ('.png', '.webp', '.tiff', '.bmp')
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"⚠️ No images found in '{INPUT_FOLDER}'")
        return

    print(f"🚀 Processing {len(files)} images to {TARGET_SIZE}x{TARGET_SIZE}...\n")

    for filename in files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGBA")
                
                # 1. Remove the empty transparent borders
                bbox = img.getbbox()
                if not bbox:
                    print(f"⚠️ Skipped: {filename} (Empty)")
                    continue
                
                cropped = img.crop(bbox)

                # 2. Calculate aspect ratio to resize without stretching
                # We scale the icon so its longest side matches TARGET_SIZE
                width, height = cropped.size
                aspect_ratio = width / height

                if width > height:
                    new_width = TARGET_SIZE
                    new_height = int(TARGET_SIZE / aspect_ratio)
                else:
                    new_height = TARGET_SIZE
                    new_width = int(TARGET_SIZE * aspect_ratio)

                # Use Resampling.LANCZOS for high-quality downscaling
                resized_icon = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # 3. Create a blank square transparent canvas and paste the icon in the center
                final_image = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0))
                
                # Calculate centering coordinates
                offset_x = (TARGET_SIZE - new_width) // 2
                offset_y = (TARGET_SIZE - new_height) // 2
                
                final_image.paste(resized_icon, (offset_x, offset_y))
                
                # 4. Save
                final_image.save(output_path)
                print(f"✅ Processed: {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"\n✨ Done! Check the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    if os.path.exists(INPUT_FOLDER):
        process_icons()
    else:
        print(f"❌ Error: Folder '{INPUT_FOLDER}' not found.")