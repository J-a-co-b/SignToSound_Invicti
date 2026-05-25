import os

# Base path
base_path = os.path.join(os.path.expanduser("~"), "Downloads", "American Sign Language Dataset", "American Sign Language Dataset")

def count_files(folder_name):
    path = os.path.join(base_path, folder_name)
    if not os.path.exists(path):
        return "❌ Not Found", 0, False
    
    items = os.listdir(path)
    # Check if inside is folders (A, B, C) or files
    if len(items) > 0 and os.path.isdir(os.path.join(path, items[0])):
        # It's a folder of folders! Count files inside the first folder (e.g., inside 'A')
        first_cat = items[0]
        sub_files = os.listdir(os.path.join(path, first_cat))
        return f"✅ Found {len(items)} categories (Folders). Example '{first_cat}' has {len(sub_files)} files.", len(sub_files), True
    else:
        return f"⚠️ Found {len(items)} files (Flat structure).", len(items), False

# Check 'Augmented Data'
print(f"--- Checking 'Augmented Data' ---")
msg, count_aug, is_folders_aug = count_files("Augmented Data")
print(msg)

# Check 'Skeleton Data'
print(f"\n--- Checking 'Skeleton Data' ---")
msg, count_skel, is_folders_skel = count_files("Skeleton Data")
print(msg)