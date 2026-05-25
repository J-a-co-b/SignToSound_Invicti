import os
import cv2
import matplotlib.pyplot as plt

# --- 1. SETUP PATHS ---
# This points to: C:/Users/annad/Downloads/American Sign Language Dataset/American Sign Language Dataset
base_path = os.path.join(os.path.expanduser("~"), "Downloads", "American Sign Language Dataset", "American Sign Language Dataset")

# We want to look inside 'Root Images' or 'Augmented Data'
images_path = os.path.join(base_path, "Root Images")

print(f"Checking for data at: {images_path}")

# --- 2. VERIFY DATA ---
if not os.path.exists(images_path):
    print("\n❌ Error: Could not find 'Root Images' folder.")
    print("Check if you renamed the folder in Downloads.")
else:
    print("\n✅ Found 'Root Images' folder!")
    
    # List the categories (e.g., A, B, C...)
    categories = os.listdir(images_path)
    print(f"Found {len(categories)} categories: {categories[:5]} ...") # Show first 5

    # --- 3. SHOW ONE IMAGE ---
    # Let's peek at the first image in the first category to make sure it works
    first_category = categories[0]
    cat_path = os.path.join(images_path, first_category)
    
    if os.path.isdir(cat_path):
        image_files = os.listdir(cat_path)
        if len(image_files) > 0:
            first_image = image_files[0]
            img_path = os.path.join(cat_path, first_image)
            
            # Read and show the image
            img = cv2.imread(img_path)
            if img is not None:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"Sample Image from Category: {first_category}")
                plt.axis('off')
                plt.show()
                print("✅ Successfully opened an image! We are ready to train.")
            else:
                print("❌ Found file but could not read image.")
        else:
            print(f"❌ Folder '{first_category}' is empty.")