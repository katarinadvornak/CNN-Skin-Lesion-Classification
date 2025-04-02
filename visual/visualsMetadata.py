import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loadmetadata import MetadataLoader
from loadimages import ImageLoader


metadata = pd.read_csv("/Users/katarinadvornak/team10/HAM10000_metadata.csv", delimiter=",")

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

# plot 5: localization and disease matrix
pivot_table = metadata.pivot_table(index="localization", columns="dx", aggfunc="size", fill_value=0)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, cmap="Blues", fmt="d")
plt.title("Disease Occurrence by Localization")
plt.xlabel("Disease Type")
plt.ylabel("Body Part (Localization)")
plt.xticks(rotation=45)
plt.show()

