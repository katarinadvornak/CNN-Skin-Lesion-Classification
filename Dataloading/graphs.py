import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loadmetadata import MetadataLoader
from loadimages import ImageLoader

metadata_loader = MetadataLoader("HAM10000_metadata")
metadata = metadata_loader.load_metadata()

print(f"metadata loading complete\n")
print(metadata.head(5))


image_loader = ImageLoader(
 "/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1", 
 "/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2", 
)


try:
    images, image_ids = image_loader.load_images()
    print("Successfully loaded", len(images), "images.")
except Exception as e:
    print("Error loading images:", str(e))


print(f"Loaded {len(images)} images with corresponding labels.")
print(f"image ids: {image_ids}")


#splitting the data depending on diseases
grouped_data = metadata.groupby('dx')

for disease, group in grouped_data:
    print(f"Group: {disease}")
    print()

# Optionally, you can also save each group as a separate CSV file if needed
for disease, group in grouped_data:
    group.to_csv(f'{disease}_group.csv', index=False)  # Save each group as a CSV

import matplotlib.pyplot as plt

# plotting the data
plt.figure(figsize=(12,12))

# plot 1: Disease distribution - histogram 
plt.subplot(2, 2, 1)
disease_counts = metadata['dx'].value_counts()
disease_counts.plot(kind='bar', title='Disease Distribution', color='skyblue')
plt.xlabel('Disease')
plt.ylabel('Count')

# plot 2: Age distribution - histogram
plt.subplot(2, 2, 2)
plt.hist(metadata['age'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# plot 3: Age distribution by disease - boxplot
plt.subplot(2, 2, 3)
sns.boxplot(x='dx', y='age', data=metadata, palette='Set2')
plt.title('Age Distribution by Disease')
plt.xlabel('Disease', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.xticks(rotation=90)


# plot 4: sex and localization distribution
plt.subplot(2, 2, 4)
sns.countplot(x='localization', hue='sex', data=metadata, palette='Set2')
plt.title('Location Distribution by Sex')
plt.xlabel('Disease', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.xticks(rotation=90)


plt.tight_layout()

plt.show()









