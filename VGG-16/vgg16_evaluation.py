import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Define paths
base_path = "/content/drive/MyDrive/project2-2/data/data"
model_path = os.path.join(base_path, "vgg16_finetuned.keras")
test_csv = os.path.join(base_path, "test/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")
test_img_dir = os.path.join(base_path, "test/ISIC2018_Task3_Test_Input")
encoder_path = os.path.join(base_path, "label_encoder.pkl")

# Load label encoder
with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)
ordered_class_names = label_encoder.classes_

# Load test labels
test_df = pd.read_csv(test_csv)
test_df.columns = [col.lower() for col in test_df.columns]
test_df["label"] = test_df[ordered_class_names].values.argmax(axis=1)
test_df["path"] = test_df["image"].apply(lambda x: os.path.join(test_img_dir, x + ".jpg"))

# Load and preprocess test images
X_test, y_test = [], []
for _, row in test_df.iterrows():
    try:
        img = load_img(row["path"], target_size=(128, 128))
        img = img_to_array(img) / 255.0
        X_test.append(img)
        y_test.append(row["label"])
    except Exception as e:
        print(f"Failed to load {row['path']}: {e}")

X_test = np.array(X_test)
y_test = np.array(y_test)
y_test_cat = to_categorical(y_test, num_classes=len(ordered_class_names))

# Load trained model
model = load_model(model_path)

# Evaluate
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n Test Accuracy: {acc:.4f}")

# Classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=ordered_class_names))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
