from loadimages import ImageLoader
from neuralnetwork import prepare_data, build_model
import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Define the paths
image_folder1 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1'
image_folder2 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2'
metadata_path = 'HAM10000_metadata'

image_loader = ImageLoader(
 image_folder1, 
 image_folder2, 
 metadata_path
)

# loading images
try:
    images, image_labels, image_ids = image_loader.load_images()
    print("Successfully loaded", len(images), "images.")
except Exception as e:
    print("Error loading images:", str(e))


# counting rows/instances in metadata to see if it matches the number of images
try:
    numberrows = image_loader.count_rows()
    print("There are", numberrows, "rows in metadata")
    image_loader.print_rows(2)

except Exception as e:
    print("Error loading images:", str(e))
    exit()




# Prepare dataset (split into train/validation + one-hot encode labels)
X_train, X_val, y_train, y_val, label_encoder = prepare_data(images, image_labels)

# Build model
input_shape = X_train.shape[1:]  # Shape of one image
num_classes = y_train.shape[1]  # Number of unique disease classes
model = build_model(input_shape, num_classes)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Save the trained model
model.save('skin_disease_model.h5')

print("Training complete. Model saved as 'skin_disease_model.h5'.")

# Load the saved model
model = load_model('skin_disease_model.h5')






