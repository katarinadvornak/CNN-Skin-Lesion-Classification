import os
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

class AugmentedImageGenerator:
    def __init__(self, image_dirs, metadata_path, output_dir, use_augmentation=True, test_size=0.2, random_state=42):
        self.image_dirs = image_dirs
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.image_size = (28, 28)
        self.use_augmentation = use_augmentation
        self.test_size = test_size
        self.random_state = random_state

        self.datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_metadata(self):
        return pd.read_csv(self.metadata_path)

    def _get_image_path(self, image_id):
        for folder in self.image_dirs:
            path = os.path.join(folder, f"{image_id}.jpg")
            if os.path.exists(path):
                return path
        return None

    def generate(self):
        metadata = self._load_metadata()

        # --- Split metadata BEFORE augmentation ---
        train_df, val_df = train_test_split(
            metadata,
            test_size=self.test_size,
            stratify=metadata['dx'],
            random_state=self.random_state
        )

        print("Train class distribution (before augmentation):")
        print(train_df['dx'].value_counts())
        print("\nValidation class distribution:")
        print(val_df['dx'].value_counts())

        label_counts = train_df['dx'].value_counts()
        max_count = label_counts.max()

        retained = []
        generated = []

        for label, count in label_counts.items():
            samples = train_df[train_df['dx'] == label]
            retained.append(samples)

            needed = max_count - count
            if needed <= 0:
                continue

            print(f"Augmenting class '{label}' with {needed} samples")

            added = 0
            while added < needed:
                for _, row in samples.iterrows():
                    if added >= needed:
                        break

                    image_id = row['image_id']
                    image_path = self._get_image_path(image_id)
                    if not image_path:
                        continue

                    try:
                        img = Image.open(image_path).convert('RGB').resize(self.image_size)
                        new_id = f"{image_id}_aug{added}"
                        save_path = os.path.join(self.output_dir, f"{new_id}.jpg")

                        if self.use_augmentation:
                            x = img_to_array(img).reshape((1, *self.image_size, 3))
                            for batch in self.datagen.flow(x, batch_size=1):
                                array_to_img(batch[0]).save(save_path)
                                break
                        else:
                            img.save(save_path)

                        generated.append({'image_id': new_id, 'dx': label})
                        added += 1

                    except Exception as e:
                        print(f"Error processing {image_id}: {e}")

        # Combine and save metadata
        train_balanced = pd.concat([*retained, pd.DataFrame(generated)], ignore_index=True)

        train_csv = os.path.join(self.output_dir, "HAM10000_metadata_balanced_train.csv")
        val_csv = os.path.join(self.output_dir, "HAM10000_metadata_val.csv")

        train_balanced.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        print(f"\n Augmented training metadata saved to: {train_csv}")
        print(f" Validation metadata saved to: {val_csv}")
        print("\n Final training class distribution (after augmentation):")
        print(train_balanced['dx'].value_counts())


# === RUNNING THE GENERATOR ===
if __name__ == "__main__":
    image_dirs = [
        "/Users/jakubb/Desktop/dataverse_files/HAM10000_images_part_1",
        "/Users/jakubb/Desktop/dataverse_files/HAM10000_images_part_2"
    ]
    metadata_path = "HAM10000_metadata"
    output_dir = "/Users/jakubb/Desktop/dataverse_files/balanced"

    generator = AugmentedImageGenerator(
        image_dirs=image_dirs,
        metadata_path=metadata_path,
        output_dir=output_dir,
        use_augmentation=True
    )

    generator.generate()
