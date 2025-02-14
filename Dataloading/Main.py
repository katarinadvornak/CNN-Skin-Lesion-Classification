import numpy as np 
from tensorflow.keras.models import load_model
from predictinglogic import load_and_preprocess_image
from sklearn.preprocessing import LabelEncoder
import pickle


# Load the saved model
model = load_model('skin_disease_model.h5')

# Load the label encoder from the pickle file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Path to the new image you want to predict
image_path = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1/ISIC_0024349.jpg'

# Preprocess the image
image = load_and_preprocess_image(image_path)

# Make a prediction
predictions = model.predict(image)

# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)

predicted_label = label_encoder.inverse_transform(predicted_class)
# Print the predicted class
print("Predicted disease:", predicted_label[0])
