import numpy as np 
from tensorflow.keras.models import load_model
from predictinglogic import load_and_preprocess_image
from sklearn.preprocessing import LabelEncoder
import pickle
from training import metadata_path
import pandas as pd


# Loading the saved model
model = load_model('skin_disease_model.h5')

# Loading the label encoder from the pickle file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Path to new image to predict
image_path = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2/ISIC_0029760.jpg'
image_id = image_path.split("/")[-1].replace(".jpg", "")
image = load_and_preprocess_image(image_path)


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
