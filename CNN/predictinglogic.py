from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path)
    # Resizing
    image = image.resize(target_size)
    
    # Converting the image to a numpy array and normalizing pixel values to [0, 1]
    image = np.array(image) / 255.0
    
    # Ensuring the image has the shape (1, height, width, channels) for the model
    image = np.expand_dims(image, axis=0)
    
    return image


