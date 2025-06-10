import pickle
from loadimages import ImageLoader
from neuralnetwork_features import build_model
import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from paramtuner import ParameterTuner


#model_path = 'skin_disease_model.keras'
model_path = 'skin_disease_model644x644.features.keras' 
#model_path = 'skin_disease_model_64x64.weights.keras'

# Define the paths
image_folder1 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1'
image_folder2 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2'
metadata_path = 'cleaned_data.csv'

image_loader = ImageLoader(
image_folder1, 
image_folder2, 
metadata_path
)


if __name__ == "__main__":
    images, image_labels, image_ids, tabular_features = image_loader.load_images()
    print(f"Successfully loaded {len(images)} images.")

    # Prepare labels
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(image_labels)
    one_hot_labels = to_categorical(integer_labels)

    # Convert to numpy arrays
    images = np.array(images)
    image_ids = np.array(image_ids)

    # Train-validation split
    X_train, X_val, X_train_tab, X_val_tab, y_train, y_val, int_train, int_val, ids_train, ids_val = train_test_split(
    images,
    tabular_features,
    one_hot_labels,
    integer_labels,
    image_ids,
    test_size=0.2,
    random_state=42
)

        # Compute class weights
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(int_train),
        y=int_train
    )
    class_weights = dict(enumerate(class_weights_array))

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # ----- ANALYSIS: Check label distribution and weights -----
    # Count the original labels before encoding
    label_counts = Counter(image_labels)
    print("\nClass distribution (raw counts):")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # Print weights
    print("\nClass weights (used during training):")
    for i, weight in class_weights.items():
        class_name = label_encoder.inverse_transform([i])[0]
        print(f"{class_name} (class {i}): {weight:.4f}")

    # Optional: Visualize frequency vs weight
    class_names = [label_encoder.inverse_transform([i])[0] for i in class_weights.keys()]
    frequencies = [label_counts[cls] for cls in class_names]
    weights = [class_weights[i] for i in class_weights.keys()]
    x = np.arange(len(class_names))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(x, frequencies)
    plt.xticks(x, class_names, rotation=45)
    plt.title("Class Frequency")
    plt.xlabel("Disease Class")
    plt.ylabel("Number of Images")

    plt.subplot(1, 2, 2)
    plt.bar(x, weights)
    plt.xticks(x, class_names, rotation=45)
    plt.title("Class Weights (Higher = Rarer)")
    plt.xlabel("Disease Class")
    plt.ylabel("Weight Value")

    plt.tight_layout()
    plt.show()
    # -----------------------------------------------------------


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


    # Either load or build model
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("Building new model...")
        input_shape = X_train.shape[1:]
        tabular_dim = X_train_tab.shape[1]
        num_classes = y_train.shape[1]
        model = build_model(input_shape, tabular_dim, num_classes)


    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


    #print("Starting hyperparameter tuning")
    #input_shape = X_train.shape[1:]
    #num_classes = y_train.shape[1]

    # Initialize and run the tuning process
    #tuner = ParameterTuner(input_shape, num_classes)
    #model, best_hps = tuner.run_tuning(X_train, y_train, X_val, y_val)
    #tuner.export_results()


    # Train the model
    model.fit(
        [X_train, X_train_tab],
        y_train,
        validation_data=([X_val, X_val_tab], y_val),
        batch_size=16,
        epochs=140,
        shuffle=True,
        class_weight=class_weights
)


    # Save the model
    model.save(model_path)
    print(f"Training complete. Model saved to '{model_path}'.")
 
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)




