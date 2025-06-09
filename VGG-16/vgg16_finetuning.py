import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Define paths
base_path = "/content/drive/MyDrive/project2-2/data/data"
model_path = os.path.join(base_path, "vgg16_skin_model.keras")
finetuned_model_path = os.path.join(base_path, "vgg16_finetuned.keras")

# Load train/val dataframes and label encoder
metadata_path = os.path.join(base_path, "train/HAM10000_metadata.csv")
with open(os.path.join(base_path, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# Load metadata and map paths
train_dirs = [
    os.path.join(base_path, "train/HAM10000_images_part_1"),
    os.path.join(base_path, "train/HAM10000_images_part_2")
]
df = pd.read_csv(metadata_path)
image_map = {}
for d in train_dirs:
    for f in os.listdir(d):
        if f.endswith(".jpg"):
            image_map[f.split(".")[0]] = os.path.join(d, f)
df["path"] = df["image_id"].map(image_map)
df = df[df["path"].notna()]
df["label"] = label_encoder.transform(df["dx"])

# Split dataset
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=42)

# Convert labels to string for generators
train_df["label"] = train_df["label"].astype(str)
val_df["label"] = val_df["label"].astype(str)

# Manual class weights
manual_class_weights = {0: 1.00, 1: 4.39, 2: 4.45, 3: 7.73, 4: 10.92, 5: 18.79, 6: 21.77}

# Image generators
img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    class_mode='sparse',
    batch_size=batch_size,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    class_mode='sparse',
    batch_size=batch_size,
    shuffle=False
)

# Load the best frozen model
model = load_model(model_path)

# Unfreeze last three convolutional blocks
for layer in model.layers:
    if any(block in layer.name for block in ['block5', 'block4', 'block3']):
        layer.trainable = True

# Recompile the model
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks for fine-tuning
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=finetuned_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Continue training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=manual_class_weights,
    callbacks=callbacks
)
