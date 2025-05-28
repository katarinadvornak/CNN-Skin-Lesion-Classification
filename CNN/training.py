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
from tensorflow.keras.callbacks import EarlyStopping

#model_path = 'skin_disease_model.keras'
model_path = 'skin_disease_model128x128.keras' 

# Define the paths
image_folder1 = '/Users/jakubb/Desktop/dataverse_files/HAM10000_images_part_1'
image_folder2 = '/Users/jakubb/Desktop/dataverse_files/HAM10000_images_part_2'
metadata_path = 'HAM10000_metadata'

image_loader = ImageLoader(
image_folder1, 
image_folder2, 
metadata_path
)

if __name__ == "__main__":
    images, image_labels, image_ids = image_loader.load_images()
    print(f"Successfully loaded {len(images)} images.")

     # Prepare labels
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(image_labels)
    one_hot_labels = to_categorical(integer_labels)

    # Convert to numpy arrays
    images = np.array(images)
    image_ids = np.array(image_ids)

    # Train-validation split
    X_train, X_val, y_train, y_val, int_train, int_val, ids_train, ids_val = train_test_split(
        images, one_hot_labels, integer_labels, image_ids,
        test_size=0.2,
        random_state=42
    )  


    # counting rows/instances in metadata to see if it matches the number of images
    #try:
    #   numberrows = image_loader.count_rows()
    #   print("There are", numberrows, "rows in metadata")
    #    image_loader.print_rows(2)
    #except Exception as e:
    #   print("Error loading images:", str(e))
    #   exit()


    # Either load or build model
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("Building new model...")
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]
        model = build_model(input_shape, num_classes)

    early_stop = EarlyStopping(
        monitor='val_loss',        # Also possible to use 'val_accuracy'
        patience=5,                # Number of epochs with no improvement before stopping
        restore_best_weights=True  # Restores weights from the best epoch
    )

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Here we adjust epochs for early stoppping
        batch_size=32,
        shuffle=True,
        callbacks=[early_stop]
    )

    # Save the model
    model.save(model_path)
    print(f"Training complete. Model saved to '{model_path}'.")

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    #print("Test image IDs:")
    #for img_id in test_ids:
    #    print(img_id)




