import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

def prepare_data(images, labels):
    # Converting labels (diseases) to categorical format (numbers: for example, 
    # [disease1, disease2, disease3] will be [0,1,2])
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Save the LabelEncoder to a file after fitting
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

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
        # 32 filters of size 3×3 will be applied and each filter slides across the image and produces one feature map.
        # Since we have 32 filters, we get 32 output feature maps
        # step by step what happens: 
        #
         #   1. Each of the 32 filters slides over the input image.
         #   2. Each filter extracts a different feature (e.g., edges, corners, textures).
         #   3. The output is 32 feature maps, stacked together.
         #   4. The activation function (like ReLU) is applied to introduce non-linearity.
         #   5. The feature maps are passed to the next layer (e.g., pooling or another convolutional layer).
        #

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # After each convolutional layer we use MaxPooling layers. 
        # They reduce the size of the data (height and width) coming from the convolutional layers. 
        # The output is smaller in size, but the most important features from th original image stays.
        # Max-pooling takes a small window (usually 2x2 or 3x3) and slides it across the image, selecting the maximum value in each window
        # No learnable parameters, jthey just reduce computation time and avoid overfitting
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # flatten means that we reshape the 2D vetor of images to a 1D vector 
        layers.Flatten(),

        # A Dense layer in TensorFlow/Keras is a fully connected (FC) layer, 
        # Meaning every neuron in the layer is connected to every neuron in the previous layer. 
        # The dense layer computes this formula: Output = Activation(W*X + b)
        # W = weights
        # X = input from previous layer
        # b = Bias

        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')

    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

