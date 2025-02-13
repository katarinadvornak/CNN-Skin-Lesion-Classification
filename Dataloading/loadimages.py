import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split


class ImageLoader:
    def __init__(self, imagespart1, imagespart2, metadata_loader):
        self.imagespart1 = imagespart1
        self.imagespart2 = imagespart2

    def load_image(self, image_path, image_size=(224, 224)):
        image = Image.open(image_path)
        image = image.resize(image_size)
        image = np.array(image)
        image = image / 255.0  # Normalize pixel values to [0, 1]
        return image

    def load_images(self):
        images = []
        image_ids = []
        labels = []

        # Load images from part 1 and part 2, and assign labels based on image_id
        for folder in [self.imagespart1, self.imagespart2]:
            for filename in os.listdir(folder):
                image_path = os.path.join(folder, filename)
                if image_path.endswith(('.jpg', '.png', '.jpeg')):
                    image_id = filename.split('.')[0]  # Extract the image ID
                    image = self.load_image(image_path)
                    disease = self.metadata_loader.get_disease_label(image_id)  # Get the disease label
                    images.append(image)
                    image_ids.append(image_id)
                    labels.append(disease)

        return np.array(images), np.array(labels), image_ids    




