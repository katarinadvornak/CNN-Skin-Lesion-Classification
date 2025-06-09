import pickle
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from loadimages import ImageLoader
from neuralnetwork import build_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION --
model_path = '' # Here specify the path to save the model eg. skin_disease_model.keras
image_folder1 = '/Users/jakubb/Desktop/dataverse_files/HAM10000_images_part_1' # Path to the first image folder
image_folder2 = '/Users/jakubb/Desktop/dataverse_files/HAM10000_images_part_2' # Path to the second image folder
metadata_path = 'HAM10000_metadata' # Path to the metadata CSV file

# If the model path is not specified, it will create a new model in the current working directory
if not model_path:
    model_dir = os.getcwd()
    model_filename = 'weights28.keras' 
    model_path = os.path.join(model_dir, model_filename)

os.makedirs(os.path.dirname(model_path), exist_ok=True)

# --- Load images and labels ---
image_loader = ImageLoader(image_folder1, image_folder2, metadata_path)
images, image_labels, image_ids = image_loader.load_images()
print(f"Successfully loaded {len(images)} images.")

# --- Encode labels ---
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(image_labels)
one_hot_labels = to_categorical(integer_labels)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# --- Prepare data ---
images = np.array(images)
image_ids = np.array(image_ids)

X_train, X_val, y_train, y_val, int_train, int_val, ids_train, ids_val = train_test_split(
    images, one_hot_labels, integer_labels, image_ids,
    test_size=0.2,
    random_state=42
)

# --- Compute class weights automatically ---
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(int_train),
    y=int_train
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

print("\nComputed class weights:")
for label, weight in zip(label_encoder.classes_, class_weights_array):
    print(f"{label}: {weight:.2f}")

# --- Build or load model ---
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Building new model...")
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes)

# --- Early stopping ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# --- Train model ---
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    batch_size=32,
    shuffle=True,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# --- Save model ---
model.save(model_path)
print(f"Training complete. Model saved to '{model_path}'.")

# --- Plot training curves ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# --- Plot class weights ---
weights = list(class_weights.values())
classes = label_encoder.classes_

plt.figure(figsize=(8, 5))
plt.bar(classes, weights)
plt.xlabel('Disease Class')
plt.ylabel('Class Weight')
plt.title('Automatically Computed Class Weights')
plt.show()

# --- Predict on validation set ---
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

# --- Decode labels ---
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# --- Classification report ---
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# --- Accuracy score ---
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# --- Confusion matrix ---
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
