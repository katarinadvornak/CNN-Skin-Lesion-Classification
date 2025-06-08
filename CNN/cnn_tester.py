# run_cnn_tester.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from predictinglogic import load_and_preprocess_image  # Make sure this is accessible
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def extract_tabular_features_for_test(image_ids, csv_path):
    df = pd.read_csv(csv_path)
    df = df.set_index("image_id")
    ordered = df.loc[image_ids]

    # Load the preprocessor that was fitted during training
    with open('tabular_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Apply transformation
    return preprocessor.transform(ordered)


class CNNTester:
    def __init__(self, model_path, encoder_path, folder_path, ground_truth_csv=None, image_size=(64, 64)):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.folder_path = folder_path
        self.ground_truth_csv = ground_truth_csv
        self.image_size = image_size

            # Optional: Load tabular data if available
        tabular_features = None
        if self.ground_truth_csv and os.path.exists(self.ground_truth_csv):
            try:
                tabular_features = extract_tabular_features_for_test(image_ids, self.ground_truth_csv)
            except Exception as e:
                print(f"Warning: Could not extract tabular features. Proceeding without them. Error: {e}")


        self.model = load_model(self.model_path)
        with open(self.encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.evaluate()

    def load_images_from_folder(self, folder_path):
        images = []
        image_ids = []

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, fname)
                try:
                    image = load_and_preprocess_image(image_path, target_size=self.image_size)
                    images.append(image)
                    image_ids.append(os.path.splitext(fname)[0])
                except Exception as e:
                    print(f"Could not process {fname}: {e}")

        if not images:
            raise ValueError("No valid images found in folder.")

        images = np.vstack(images)
        return images, image_ids


    def evaluate(self):
        print(f"\nEvaluating on folder: {self.folder_path}")
        images, image_ids = self.load_images_from_folder(self.folder_path)

        # Try to extract tabular features if model expects them
        tabular_features = None
        if len(self.model.inputs) == 2:
            try:
                tabular_features = extract_tabular_features_for_test(image_ids, self.ground_truth_csv)
            except Exception as e:
                print(f"⚠️ Could not extract tabular features. Model expects 2 inputs. Error: {e}")
                return  # Exit to avoid model input mismatch

        print("Running predictions...")
        if tabular_features is not None:
            preds = self.model.predict([images, tabular_features])
        else:
            preds = self.model.predict(images)

        pred_classes = np.argmax(preds, axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_classes)

        # Evaluation
        if self.ground_truth_csv and os.path.exists(self.ground_truth_csv):
            gt_df = pd.read_csv(self.ground_truth_csv)
            pred_df = pd.DataFrame({'image_id': image_ids, 'predicted': pred_labels})
            merged = pd.merge(gt_df, pred_df, on='image_id')

            y_true = merged['dx']
            y_pred = merged['predicted']

            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))

            print("Confusion Matrix:")
            self._plot_confusion_matrix(y_true, y_pred)

            # Class distribution plot
            print("\nClass distribution in test set (visualized):")
            class_counts = y_true.value_counts().sort_index()
            plt.figure(figsize=(8, 5))
            sns.barplot(x=class_counts.index, y=class_counts.values)
            plt.xlabel('Class')
            plt.ylabel('Number of Images')
            plt.title('Test Set Class Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        else:
            print("\nPredictions (no ground truth available):")
            for img_id, label in zip(image_ids, pred_labels):
                print(f"{img_id}: {label}")

    

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=self.label_encoder.classes_)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()


    

if __name__ == "__main__":
    tester = CNNTester(
        model_path='skin_disease_model644x644.features.keras',
        encoder_path='label_encoder.pkl',
        folder_path='/Users/ninazorawska/Desktop/project 22/ISIC2018_Task3_Test_Images',
        ground_truth_csv='/Users/ninazorawska/Desktop/project 22/ISIC2018_Task3_Test_GroundTruth.csv'
    )
