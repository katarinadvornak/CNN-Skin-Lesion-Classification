from PIL import Image
import numpy as np
import os

imagespart1 = "/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1"
imagespart2 = "/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2 (1)"
image_id = "0029307"
image_filename = "ISIC_{image_id}.jpg"


def load_image(image_path, image_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Initialize lists to store images and their labels
images = []
labels = []

# Load images from folder1 and assign label 0
for filename in os.listdir(imagespart1):
    image_path = os.path.join(imagespart1, filename)
    if image_path.endswith(('.jpg', '.png', '.jpeg')):  # Ensure the file is an image
        image = load_image(image_path)
        images.append(image)
        labels.append(0)  # Label 0 for images in folder1

# Load images from folder2 and assign label 1
for filename in os.listdir(imagespart2):
    image_path = os.path.join(imagespart2, filename)
    if image_path.endswith(('.jpg', '.png', '.jpeg')):  # Ensure the file is an image
        image = load_image(image_path)
        images.append(image)
        labels.append(1)  # Label 1 for images in folder2

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Example output
print(f"Loaded {len(images)} images with corresponding labels.")


