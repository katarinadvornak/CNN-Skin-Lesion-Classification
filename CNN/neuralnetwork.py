import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras import layers, models, optimizers, Input, Model
from tensorflow.keras.layers import concatenate

def build_model(image_shape, tabular_input_dim, num_classes, activation='relu', dropout_rate=0.3, learning_rate=1e-4):
    # Image input branch
    image_input = Input(shape=image_shape, name="image_input")
    x = layers.Conv2D(32, (3, 3), padding='same')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Tabular input branch
    tabular_input = Input(shape=(tabular_input_dim,), name="tabular_input")
    t = layers.Dense(64, activation=activation)(tabular_input)
    t = layers.Dropout(dropout_rate)(t)

    # Combine both branches
    combined = concatenate([x, t])
    combined = layers.Dense(128, activation=activation)(combined)
    combined = layers.Dropout(dropout_rate)(combined)

    output = layers.Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[image_input, tabular_input], outputs=output)

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

# Function to build the CNN model with tunable hyperparameters for Keras Tuner
def build_tunable_model(hp):
    model = models.Sequential()

    # input layer
    model.add(layers.InputLayer(input_shape=(64, 64, 3)))  # default fixed input shape

    # Tune number of convolutional layers (between 2 and 3)
    for i in range(hp.Int("conv_layers", 1, 5)):
        kernel_size = hp.Choice(f"conv_{i}_kernel_size", values=[3, 5])
        activation = hp.Choice(f"conv_{i}_activation", values=["relu", "tanh", "sigmoid", "selu"])
        # Each of the conv layers will have a tunable number of filters
        model.add(layers.Conv2D(
            filters=hp.Choice(f"conv_{i}_filters", values=[32, 64, 128]),
            kernel_size=(kernel_size, kernel_size),
            activation=activation,
            padding='same'
        ))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())


    # Tune the number of neurons in the dense layer
    model.add(layers.Dense(
        units=hp.Int("dense_units", min_value=128, max_value=1152, step=128),
        activation='relu'
    ))

    # Output layer (7 disease classes from your dataset)
    model.add(layers.Dense(
        units=hp.Int("num_classes", 7, 7),  # fixed to 7
        activation='softmax'
    ))

    # Compile the model with tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

