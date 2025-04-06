import pickle
from loadimages import ImageLoader
from neuralnetwork import build_model
import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from splitting import DataSplitter  # or wherever you put it

#model_path = 'skin_disease_model.keras'
model_path = 'skin_disease_model128x128.keras' 

# Define the paths
image_folder1 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1'
image_folder2 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2'
metadata_path = 'HAM10000_metadata'

image_loader = ImageLoader(
image_folder1, 
image_folder2, 
metadata_path
)

if __name__ == "__main__":
    images, image_labels, image_ids = image_loader.load_images()
    print(f"Successfully loaded {len(images)} images.")

    # Split data
    splitter = DataSplitter(val_size=0.2, test_size=0.1)
    X_train, X_val, y_train, y_val, X_test, y_test, test_ids, label_encoder = splitter.split(images, image_labels, image_ids)

    # Save test set if you want to use it later:
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # X_train - The images used for training the CNN
    # X_val - The images used to validate/test the model
    # y_train - One-hot encoded labels (diseases) for the training images
    # y_val - The one-hot encoded lables for the validation images
    # label_encoder - the converter that converts disease names into numbers, that we use for splitting and predicting.

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)



    # counting rows/instances in metadata to see if it matches the number of images
    #try:
    #   numberrows = image_loader.count_rows()
    #   print("There are", numberrows, "rows in metadata")
    #    image_loader.print_rows(2)
    #except Exception as e:
    #   print("Error loading images:", str(e))
    #   exit()


    # Saving the encoder for the disease lables, which we used in splitting the data, so we can use the same one in predicting the disease.
    splitter.save_encoder('label_encoder.pkl')

    # Either load or build model
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("Building new model...")
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]
        model = build_model(input_shape, num_classes)

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32, shuffle=True)

    # Save the model
    model.save(model_path)
    print(f"Training complete. Model saved to '{model_path}'.")

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    print("Test image IDs:")
    for img_id in test_ids:
        print(img_id)




