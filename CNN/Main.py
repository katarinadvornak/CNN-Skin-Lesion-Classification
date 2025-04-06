import numpy as np 
from tensorflow.keras.models import load_model
from predictinglogic import load_and_preprocess_image
from sklearn.preprocessing import LabelEncoder
import pickle
from training import metadata_path
import pandas as pd
import os

# Loading the saved model
model = load_model('skin_disease_model128x128.keras')

# Loading the label encoder from the pickle file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


def load_image_by_id(image_id, folder1, folder2):
    # Construct possible full paths
    filename = f"{image_id}.jpg"
    path1 = os.path.join(folder1, filename)
    path2 = os.path.join(folder2, filename)

    # Check which one exists
    if os.path.exists(path1):
        return load_and_preprocess_image(path1)
    elif os.path.exists(path2):
        return load_and_preprocess_image(path2)
    else:
        raise FileNotFoundError(f"Image {filename} not found in either folder.")

image_id = "ISIC_0025957"
folder1 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1'
folder2 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2'
image = load_image_by_id(image_id, folder1, folder2)


predictions = model.predict(image)

#the class with the largest probability
predicted_class = np.argmax(predictions, axis=1)

predicted_label = label_encoder.inverse_transform(predicted_class)
print("Predicted disease:", predicted_label[0])

# checking if the disease is correct:
df = pd.read_csv(metadata_path)
correctlabelrow = df[df.iloc[:, 1] == image_id] 
if not correctlabelrow.empty:
    correct_label = correctlabelrow.iloc[0, 2] 
else:
    correct_label = "Unknown"

print("Correct disease:", correct_label)
