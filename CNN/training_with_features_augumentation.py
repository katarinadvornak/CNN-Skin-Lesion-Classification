import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from loadimages import ImageLoader
from neuralnetwork import build_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class CNNTrainer:
    def __init__(self,
                 image_folder,
                 train_metadata_path,
                 val_metadata_path,
                 model_path='skin_disease_model64x64.featuresaugumented.keras',
                 image_size=(64, 64)):

        self.image_folder = image_folder
        self.train_metadata_path = train_metadata_path
        self.val_metadata_path = val_metadata_path
        self.model_path = model_path
        self.image_size = image_size

        self.label_encoder = LabelEncoder()

        self._load_data()
        self._prepare_labels()
        self._compute_class_weights()
        self._build_model()

    def _load_data(self):
            # Create preprocessor ONCE
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['age']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['sex', 'localization'])
        ])

        # Create loader and fit preprocessor
        train_loader = ImageLoader(
            self.image_folder,
            self.train_metadata_path,
            image_size=self.image_size,
            preprocessor=self.preprocessor,
            fit_preprocessor=True  # 👈 FIT here
        )
        self.X_train, self.y_train, self.id_train, self.X_train_tab = train_loader.load_images()

        # Reuse loader for validation (no refitting)
        val_loader = ImageLoader(
            self.image_folder,
            self.val_metadata_path,
            image_size=self.image_size,
            preprocessor=self.preprocessor,
            fit_preprocessor=False  # 👈 Just transform
        )
        self.X_val, self.y_val, self.id_val, self.X_val_tab = val_loader.load_images()

    def _prepare_labels(self):
        all_labels = np.concatenate([self.y_train, self.y_val])
        self.label_encoder.fit(all_labels)

        self.int_train = self.label_encoder.transform(self.y_train)
        self.int_val = self.label_encoder.transform(self.y_val)
        self.y_train_encoded = to_categorical(self.int_train)
        self.y_val_encoded = to_categorical(self.int_val)

        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

    def _compute_class_weights(self):
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.int_train),
            y=self.int_train
        )
        self.class_weights = dict(enumerate(class_weights_array))

        print("\nClass distribution (raw counts):")
        label_counts = Counter(self.y_train)
        for label, count in label_counts.items():
            print(f"{label}: {count}")

        print("\nClass weights (used during training):")
        for i, weight in self.class_weights.items():
            class_name = self.label_encoder.inverse_transform([i])[0]
            print(f"{class_name} (class {i}): {weight:.4f}")

        self._plot_class_distribution(label_counts)

    def _plot_class_distribution(self, label_counts):
        class_names = [self.label_encoder.inverse_transform([i])[0] for i in self.class_weights.keys()]
        frequencies = [label_counts[cls] for cls in class_names]
        weights = [self.class_weights[i] for i in self.class_weights.keys()]
        x = np.arange(len(class_names))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(x, frequencies)
        plt.xticks(x, class_names, rotation=45)
        plt.title("Class Frequency")

        plt.subplot(1, 2, 2)
        plt.bar(x, weights)
        plt.xticks(x, class_names, rotation=45)
        plt.title("Class Weights")

        plt.tight_layout()
        plt.show()



    def _build_model(self):
        if os.path.exists(self.model_path):
            print("Loading existing model...")
            self.model = load_model(self.model_path)
        else:
            print("Building new model...")
            input_shape = self.X_train.shape[1:]
            tabular_dim = self.X_train_tab.shape[1]
            num_classes = self.y_train_encoded.shape[1]
            print("Tabular input shape:", self.X_train_tab.shape, self.X_val_tab.shape)
            self.model = build_model(input_shape, tabular_dim, num_classes)

    def train(self, batch_size=16, epochs=30):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.model.fit(
            [self.X_train, self.X_train_tab],
            self.y_train_encoded,
            validation_data=([self.X_val, self.X_val_tab], self.y_val_encoded),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            class_weight=self.class_weights,
            callbacks=[early_stop]
        )

        self.model.save(self.model_path)
        print(f"\n Training complete. Model saved to '{self.model_path}'")




if __name__ == '__main__':
    trainer = CNNTrainer(
        image_folder='/Users/ninazorawska/Desktop/project 22/dataverse_files/balanced',
        train_metadata_path='HAM10000_metadata_balanced_train.csv',
        val_metadata_path='HAM10000_metadata_val.csv',
        model_path='skin_disease_model644x644.featuresaugumented.keras'
    )
    trainer.train(epochs=10)