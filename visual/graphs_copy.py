import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

<<<<<<< HEAD:visual/graphs.py

metadata_path = "HAM10000_metadata"
metadata = pd.read_csv(metadata_path)

print(f"metadata loading complete\n")
print(metadata.head(5))
=======

metadata = pd.read_csv("/Users/katarinadvornak/team10/HAM10000_metadata.csv", delimiter=",")

image_loader = ImageLoader(
    #"/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1", 
    #"/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2 (1)", 
    '/Users/katarinadvornak/Desktop/Project 2-2/data/HAM10000_images_part_1',
    '/Users/katarinadvornak/Desktop/Project 2-2/data/HAM10000_images_part_2',
   metadata
)

#Load the images and labels
images, labels = image_loader.load_images()

print(f"Loaded {len(images)} images with corresponding labels.")
>>>>>>> 2709bca (added data and cleaned it):Dataloading/loaddata.py


#splitting the data depending on diseases
grouped_data = metadata.groupby('dx')

for disease, group in grouped_data:
    print(f"Group: {disease}")
    print()

# Optionally, you can also save each group as a separate CSV file if needed
for disease, group in grouped_data:
    group.to_csv(f'{disease}_group.csv', index=False)  # Save each group as a CSV













