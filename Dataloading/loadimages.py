import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split


class ImageLoader:
    def __init__(self, imagespart1, imagespart2, metadata_path, image_size=(8, 8)):
        self.imagespart1 = imagespart1
        self.imagespart2 = imagespart2
        self.metadata_path = metadata_path
        self.metadata = None
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.image_ids = []

        self.load_metadata()

    def load_metadata(self):
        """Loads the metadata CSV into a DataFrame."""
        if self.metadata is None:  # Check if metadata is already loaded
            try:
                self.metadata = pd.read_csv(self.metadata_path)  # Load into self.metadata (not self.metadata_path)
                print("Metadata loaded successfully.")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata = None
    
    def get_disease_label(self, image_id):
        """Fetch the disease label for an image ID from the metadata."""
        disease = self.metadata.loc[self.metadata['image_id'] == image_id, 'dx'].values
        return disease[0] if disease else None

    def load_image(self, image_path):
        """Loads and preprocesses an image."""
        image = Image.open(image_path)
        image = image.resize(self.image_size)
        image = np.array(image)
        image = image / 255.0  # Normalize pixel values to [0, 1]
        return image

    def load_images(self):
        """Load images from both image folders and assign disease labels."""
        # Loop through the image folders and load images
        for folder in [self.imagespart1, self.imagespart2]:
            for filename in os.listdir(folder):
                image_path = os.path.join(folder, filename)
                if image_path.endswith(('.jpg', '.png', '.jpeg')):
                    image_id = filename.split('.')[0]  # Extract the image ID
                    image = self.load_image(image_path)

                    if image is not None:  # Only proceed if image loaded successfully
                        disease = self.get_disease_label(image_id)  # Get disease label from metadata
                        if disease is None:
                            print(f"Warning: No disease label found for image ID {image_id}")
                            continue  # Skip unlabeled images

                        self.images.append(image)
                        self.image_ids.append(image_id)
                        self.labels.append(disease)

        return np.array(self.images), np.array(self.labels), self.image_ids
    
    def count_rows(self):
        """Counts the number of rows in the metadata."""
        if self.metadata is not None:
            return len(self.metadata)
        else:
            print("Error: Metadata not loaded.")
            return 0
        
    def print_rows(self, num_rows=5):
        """Prints a specific number of rows from the metadata."""
        if self.metadata is not None:
            print(self.metadata.head(num_rows))  # Prints the first 'num_rows' rows
        else:
            print("Error: Metadata not loaded.")