from PIL import Image
import numpy as np
import os


class ImageLoader:
    def __init__(self, imagespart1, imagespart2, metadata):
        self.imagespart1 = imagespart1
        self.imagespart2 = imagespart2
        self.metadata = metadata  # Receive metadata as a parameter
        self.image_disease = {row['image_id']: row['dx'] for index, row in self.metadata.iterrows()}

    def load_image(self, image_path, image_size=(224, 224)):
        image = Image.open(image_path)
        image = image.resize(image_size)
        image = np.array(image)
        image = image / 255.0  # Normalize pixel values to [0, 1]
        return image

    def load_images(self):
        images = []
        labels = []

        # Load images from folder1 and assign labels based on the image_id
        for filename in os.listdir(self.imagespart1):
            image_path = os.path.join(self.imagespart1, filename)
            if image_path.endswith(('.jpg', '.png', '.jpeg')):  # Ensure the file is an image
                image_id = filename.split('.')[0]  # Get the image ID (without extension)
                if image_id in self.image_disease:
                    disease = self.image_disease[image_id]
                    image = self.load_image(image_path)
                    images.append(image)
                    labels.append(disease)  # Assign the disease as the label

        # Load images from folder2 and assign labels based on the image_id
        for filename in os.listdir(self.imagespart2):
            image_path = os.path.join(self.imagespart2, filename)
            if image_path.endswith(('.jpg', '.png', '.jpeg')):  # Ensure the file is an image
                image_id = filename.split('.')[0]
                if image_id in self.image_disease:
                    disease = self.image_disease[image_id]
                    image = self.load_image(image_path)
                    images.append(image)
                    labels.append(disease)  # Assign the disease as the label

        return np.array(images), np.array(labels)




