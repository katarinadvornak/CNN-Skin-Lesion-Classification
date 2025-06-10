import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle



class ImageLoader:
    def __init__(self, imagespart, metadata_path, image_size=(64, 64), preprocessor=None, fit_preprocessor=False):
        self.imagespart = imagespart
        self.metadata_path = metadata_path
        self.metadata = None
        self.image_size = image_size
        self.preprocessor = preprocessor
        self.fit_preprocessor = fit_preprocessor
        self.load_metadata()


    def load_metadata(self):
        """Loads the metadata CSV into a DataFrame."""
        if self.metadata is None: 
            try:
                self.metadata = pd.read_csv(self.metadata_path)  
                print("Metadata loaded successfully.")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata = None
    
    def get_disease_label(self, image_id):
        """Fetch the disease label for an image ID from the metadata."""
        try:
            disease = self.metadata.loc[self.metadata['image_id'] == image_id, 'dx'].values
            if len(disease) > 0:
                return disease[0]
            else:
                print(f"[DEBUG] No label found for image_id: {image_id}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to fetch disease label for image_id: {image_id} → {e}")
            return None


    def load_image(self, image_path):
        """Loads and preprocesses an image."""
        image = Image.open(image_path)
        image = image.resize(self.image_size)
        image = np.array(image)
        image = image / 255.0
        return image

    def load_images(self):
        """Load images and assign disease labels, supporting multi-folder lookup for validation."""
        # Determine if it's validation based on filename
        self.images = []
        self.labels = []
        self.image_ids = []

        is_validation = "val" in self.metadata_path.lower()
        is_training = "train" in self.metadata_path.lower()

        folders_to_search = [self.imagespart]
        
        # If it's validation, also search in the original training folders
        if is_validation or is_training:
            folders_to_search.extend([
                "/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1",
                "/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2"
            ])

       
        

        for image_id in self.metadata['image_id']:
            image_found = False
            for folder in folders_to_search:
                image_path = os.path.join(folder, f"{image_id}.jpg")
                if os.path.exists(image_path):
                    try:
                        image = self.load_image(image_path)
                        disease = self.get_disease_label(image_id)

                        if disease is not None:
                            self.images.append(image)
                            self.image_ids.append(image_id)
                            self.labels.append(disease)
                        else:
                            print(f"Warning: No label found for {image_id}")
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                    image_found = True
                    break

            if not image_found:
                print(f"Warning: Image not found in any folder: {image_id}")

        images_np = np.array(self.images)
        labels_np = np.array(self.labels)
        image_ids_np = np.array(self.image_ids)
        tabular_np = self.extract_tabular_features(image_ids_np)

        return images_np, labels_np, image_ids_np, tabular_np
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
            print(self.metadata.head(num_rows)) 
        else:
            print("Error: Metadata not loaded.")

    def extract_tabular_features(self, image_ids):
        if self.metadata is None:
            self.load_metadata()

        meta = self.metadata.set_index("image_id")
        image_ids = [img_id for img_id in image_ids if img_id in meta.index]
        ordered_meta = meta.loc[image_ids]

        categorical_cols = ['sex', 'localization']
        numeric_cols = ['age']

        if self.preprocessor is None:
            self.preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

        if self.fit_preprocessor:
            self.preprocessor.fit(ordered_meta)

        return self.preprocessor.transform(ordered_meta)
