import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loadmetadata import MetadataLoader
from loadimages import ImageLoader


metadata = pd.read_csv("/Users/katarinadvornak/team10/HAM10000_metadata.csv", delimiter=",")

# Check for null values
nan_count = metadata.isnull().sum()
print("\nMissing Values:\n")
print(nan_count)

# Check for duplicates
print("\nDuplicates:\n")
duplicate_count = metadata['image_id'].duplicated().sum
print(duplicate_count)

# Get statistics for age
print("\nAge Statistics:\n")
print(metadata['age'].describe(), "\n" + "-"*50)

# Get value counts for categorical variables
print("\nLocalization Distribution:\n")
print(metadata['localization'].value_counts(), "\n" + "-"*50)

print("\nex Distribution:\n")
print(metadata['sex'].value_counts(), "\n" + "-"*50)

print("\nDisease (dx) Distribution:\n")
print(metadata['dx'].value_counts(), "\n" + "-"*50)

# Get unique values count for categorical columns
categorical_columns = ['dx', 'dx_type', 'sex', 'localization', 'dataset']
unique_values = {col: metadata[col].nunique() for col in categorical_columns}
print("\nUnique values:\n")
print(unique_values)


