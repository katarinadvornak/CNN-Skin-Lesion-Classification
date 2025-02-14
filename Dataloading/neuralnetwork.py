import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_data(images, labels):
    # Converting labels (diseases) to categorical format (numbers: for example, 
    # [disease1, disease2, disease3] will be [0,1,2])
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # one-hot encoding on the labels, so we are transforming the numerical labels 
    # into binary vectors, so for example number 1 will be a vector: [0, 1, 0]
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_encoder.classes_))
    
    # Splitting the data into training and validation sets (80% is training, 20% is validation)
    # this means test_size is usually set to 0.2
    # X_train: The training images. X_val: The validation images. 
    # y_train: The one-hot encoded labels (vectors) for the training set. y_val: The one-hot encoded labels for the validation set.
    # label_encoder: transforms the labels (useful if we want to decode predictions back to their original labels).

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, label_encoder


# Function to build the CNN model
def build_model(input_shape, num_classes):
    model = models.Sequential([

        # input layer
        layers.InputLayer(input_shape=input_shape),
        
        # three hidden layers 
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model