from PIL import Image

# Load the image
img = Image.open("pin_overlay.png") # 200x200
width, height = img.size

# Define desired dimensions
new_width, new_height = 150, 150

# Calculate coordinates
left = (width - new_width) / 2
top = (height - new_height) / 2
right = (width + new_width) / 2
bottom = (height + new_height) / 2

# Crop the image
center_cropped_img = img.crop((left, top, right, bottom))

# Save or show
center_cropped_img.save("centered_crop.png")
center_cropped_img.show()