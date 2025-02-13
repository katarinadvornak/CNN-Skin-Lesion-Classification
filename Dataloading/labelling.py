from sklearn.model_selection import train_test_split
from loadmetadata import MetadataLoader
from loadimages import ImageLoader

class MainClass:
    def __init__(self, image_folder1, image_folder2, metadata_path):
        self.metadata_loader = MetadataLoader(metadata_path)  # Instantiate MetadataLoader
        self.image_loader = ImageLoader(image_folder1, image_folder2, self.metadata_loader)  # Pass the metadata_loader instance

    def run(self):
        # Load images, labels, and image IDs
        images, labels, image_ids = self.image_loader.load_images()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            images, labels, image_ids, test_size=0.2, random_state=42
        )

        # Return the split data
        return X_train, X_test, y_train, y_test, ids_train, ids_test
