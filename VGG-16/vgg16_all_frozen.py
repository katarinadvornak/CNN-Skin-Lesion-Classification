from google.colab import drive
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import pandas as pd
import pickle

# Mount Google Drive
drive.mount('/content/drive')

# Paths
base_path = "/content/drive/MyDrive/project2-2/data/data"
train_dirs = [
    os.path.join(base_path, "train/HAM10000_images_part_1"),
    os.path.join(base_path, "train/HAM10000_images_part_2")
]
metadata_path = os.path.join(base_path, "train/HAM10000_metadata.csv")
model_path = os.path.join(base_path, "vgg16_skin_model.keras")
encoder_path = os.path.join(base_path, "label_encoder.pkl")

# Manual class weights
manual_class_weights = {0: 1.00, 1: 4.39, 2: 4.45, 3: 7.73, 4: 10.92, 5: 18.79, 6: 21.77}

# Load metadata and image paths
df = pd.read_csv(metadata_path)
image_map = {}
for d in train_dirs:
    for f in os.listdir(d):
        if f.endswith(".jpg"):
            image_map[f.split(".")[0]] = os.path.join(d, f)
df["path"] = df["image_id"].map(image_map)
df = df[df["path"].notna()]

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["dx"])
with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)

# Train-validation split
train_df, val_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=42)

def load_images_and_labels(df, img_size=(128, 128)):
    images, labels = [], []
    for _, row in df.iterrows():
        img = load_img(row["path"], target_size=img_size)
        img = img_to_array(img) / 255.0
        images.append(img)
        labels.append(row["label"])
    return np.array(images), to_categorical(labels, num_classes=7)

X_train, y_train = load_images_and_labels(train_df)
X_val, y_val = load_images_and_labels(val_df)
y_train_int = np.argmax(y_train, axis=1)
y_val_int = np.argmax(y_val, axis=1)

# Build and compile model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Checkpoint and training
checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1)
model.fit(X_train, y_train_int, validation_data=(X_val, y_val_int), epochs=10, batch_size=32, class_weight=manual_class_weights, callbacks=[checkpoint])
