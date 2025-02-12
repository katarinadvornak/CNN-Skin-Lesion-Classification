import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# You can also load the file into a pandas DataFrame (assuming it's a CSV or tabular data)
df = pd.read_csv("HAM10000_metadata")  # This assumes it's a CSV file
print(df.head())  # Prints the first 5 rows of the DataFrame
print(df.isnull().sum())


#splitting the data depending on diseases
grouped_data = df.groupby('dx')

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
disease_counts = df['dx'].value_counts()
disease_counts.plot(kind='bar', title='Disease Distribution', color='skyblue')
plt.xlabel('Disease')
plt.ylabel('Count')

# plot 2: Age distribution - histogram
plt.subplot(2, 2, 2)
plt.hist(df['age'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# plot 3: Age distribution by disease - boxplot
plt.subplot(2, 2, 3)
sns.boxplot(x='dx', y='age', data=df, palette='Set2')
plt.title('Age Distribution by Disease')
plt.xlabel('Disease', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.xticks(rotation=90)


# plot 4: sex and localization distribution
plt.subplot(2, 2, 4)
sns.countplot(x='localization', hue='sex', data=df, palette='Set2')
plt.title('Location Distribution by Sex')
plt.xlabel('Disease', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.xticks(rotation=90)


plt.tight_layout()

plt.show()









