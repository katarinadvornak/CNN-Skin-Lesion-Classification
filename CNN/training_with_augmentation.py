import pickle
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from loadimages import ImageLoader
from neuralnetwork import build_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tensorflow.keras.callbacks import ReduceLROnPlateau

# --- CONFIGURATION ---
model_path = '' # Here specify the path to save the model eg. skin_disease_model.keras
image_folder1 = '/Users/jakubb/Desktop/dataverse_files/HAM10000_images_part_1' # Path to the first image folder
image_folder2 = '/Users/jakubb/Desktop/dataverse_files/HAM10000_images_part_2' # Path to the second image folder
image_folder3 = '/Users/jakubb/Desktop/dataverse_files/balanced' # Path to the metadata CSV file

train_metadata_path = 'HAM10000_metadata_balanced_train.csv' # Path to the training metadata CSV file 
val_metadata_path = 'HAM10000_metadata_val.csv' # Path to the validation metadata CSV file

if not model_path:
    model_dir = os.getcwd()
    model_filename = 'test_weights_augm.keras'
    model_path = os.path.join(model_dir, model_filename)

os.makedirs(os.path.dirname(model_path), exist_ok=True)

# --- Load images ---
train_loader = ImageLoader(image_folder1, image_folder2, image_folder3, train_metadata_path)
X_train, y_train_labels, _ = train_loader.load_images()
print(f"Loaded {len(X_train)} training images.")

val_loader = ImageLoader(image_folder1, image_folder2, image_folder3, val_metadata_path)
X_val, y_val_labels, _ = val_loader.load_images()
print(f"Loaded {len(X_val)} validation images.")

# --- Encode labels ---
label_encoder = LabelEncoder()
label_encoder.fit(np.concatenate((y_train_labels, y_val_labels)))

y_train_int = label_encoder.transform(y_train_labels)
y_val_int = label_encoder.transform(y_val_labels)

y_train = to_categorical(y_train_int)
y_val = to_categorical(y_val_int)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# --- Convert to numpy arrays ---
X_train = np.array(X_train)
X_val = np.array(X_val)

# --- Compute class weights ---
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weight_dict = dict(enumerate(class_weights_array))
print("\nComputed class weights:")
for idx, weight in class_weight_dict.items():
    print(f"{label_encoder.classes_[idx]}: {weight:.2f}")

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
    patience=10,
    restore_best_weights=True
)

# --- Learning rate scheduler ---
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# --- Train model with class weights ---
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    batch_size=32,
    shuffle=True,
    class_weight=class_weight_dict,
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

# --- Evaluate on validation set ---
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"\nValidation Accuracy: {accuracy:.4f}")

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
