import os
import random

# Path to the directory containing the folders
DATA_DIR = "./data"


# Loop through each folder in the DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):  # Skip non-directory files like .DS_Store
        continue
    
    images = []
    # Get the list of image files in the current folder
    for img in os.listdir(dir_path):
        if img.lower().endswith(('.png', '.jpg', '.jpeg')): 
            images.append(img)

    # Calculate the number of images to delete (75%)
    num_to_delete = int(len(images) * 0.75)

    # Randomly select images to delete
    images_to_delete = random.sample(images, num_to_delete)

    # Delete the selected images
    for img in images_to_delete:
        img_path = os.path.join(dir_path, img)
        os.remove(img_path)
        print(f"Deleted: {img_path}")

    print(f"Folder '{dir_}': Deleted {num_to_delete}/{len(images)} images.")

print("Cleanup completed!")
