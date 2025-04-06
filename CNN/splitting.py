import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np

class DataSplitter:
    def __init__(self, val_size=0.2, test_size=0.1, random_state=42):
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def encode_labels(self, labels):
        integer_labels = self.label_encoder.fit_transform(labels)
        one_hot_labels = to_categorical(integer_labels)
        return one_hot_labels

    def split(self, images, labels, image_ids):
        images = np.array(images)
        one_hot_labels = self.encode_labels(labels)
        image_ids = np.array(image_ids)

        # Step 1: Split test set
        images_temp, images_test, labels_temp, labels_test, ids_temp, ids_test = train_test_split(
            images, one_hot_labels, image_ids,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Step 2: Split train and validation from the rest
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            images_temp, labels_temp, ids_temp,
            test_size=self.val_size,
            random_state=self.random_state
        )

        return X_train, X_val, y_train, y_val, images_test, labels_test, ids_test, self.label_encoder

    def save_encoder(self, path='label_encoder.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to {path}")