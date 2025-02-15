import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


metadata_path = "HAM10000_metadata"
metadata = pd.read_csv(metadata_path)

print(f"metadata loading complete\n")
print(metadata.head(5))


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









