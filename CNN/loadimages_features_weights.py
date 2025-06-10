import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle



class ImageLoader:
    def __init__(self, imagespart1, imagespart2, metadata_path, image_size=(64, 64)):
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
        if self.metadata is None: 
            try:
                self.metadata = pd.read_csv(self.metadata_path)  
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
        image = image / 255.0
        return image

    def load_images(self):
        """Load images from both image folders and assign disease labels."""
        for folder in [self.imagespart1, self.imagespart2]:
            for filename in os.listdir(folder):
                image_path = os.path.join(folder, filename)
                if image_path.endswith(('.jpg', '.png', '.jpeg')):
                    image_id = filename.split('.')[0]  
                    image = self.load_image(image_path)

                    if image is not None:  
                        disease = self.get_disease_label(image_id) 
                        if disease is None:
                            print(f"Warning: No disease label found for image ID {image_id}")
                            continue 

                        self.images.append(image)
                        self.image_ids.append(image_id)
                        self.labels.append(disease)
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
        """Extract tabular features aligned with given image IDs."""
        # Ensure metadata is available
        if self.metadata is None:
            self.load_metadata()

        # Set index to image_id
        meta = self.metadata.set_index("image_id")

        # Reorder to match image_ids
        ordered_meta = meta.loc[image_ids]

        # Define features to extract
        categorical_cols = ['sex', 'localization']
        numeric_cols = ['age']

        # Preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        X_tabular = preprocessor.fit_transform(ordered_meta)

                # Save the fitted preprocessor for use during testing
        with open('tabular_preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)

        return X_tabular
