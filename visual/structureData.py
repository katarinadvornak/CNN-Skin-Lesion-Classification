import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define Paths
BASE_DIR = "data/"
IMAGES_1 = os.path.join(BASE_DIR, "HAM10000_images_part_1")  
IMAGES_2 = os.path.join(BASE_DIR, "HAM10000_images_part_2")  
METADATA= os.path.join(BASE_DIR, "HAM10000_metadata.csv")  
OUTPUT = os.path.join(BASE_DIR, "HAM10000_Dataset")  

# Create Train/Val/Test directories
for split in ["train", "val", "test"]:
    for class_name in pd.read_csv(METADATA)["dx"].unique():
        os.makedirs(os.path.join(OUTPUT, split, class_name), exist_ok=True)

#  Load Metadata
df = pd.read_csv(METADATA)

# Split Data into Train (70%), Validation (15%), Test (15%)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["dx"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["dx"], random_state=42)

# Function to Move Images
def move_images(df_subset, split):
    for _, row in df_subset.iterrows():
        image_filename = f"{row['image_id']}.jpg"  # Assuming images are stored as .jpg
        src_path_1 = os.path.join(IMAGES_1, image_filename)
        src_path_2 = os.path.join(IMAGES_2, image_filename)
        dst_path = os.path.join(OUTPUT, split, row["dx"], image_filename)

        # Check which directory contains the image and move it
        if os.path.exists(src_path_1):
            shutil.copy(src_path_1, dst_path)
        elif os.path.exists(src_path_2):
            shutil.copy(src_path_2, dst_path)

# Move Images into Respective Folders
move_images(train_df, "train")
move_images(val_df, "val")
move_images(test_df, "test")


