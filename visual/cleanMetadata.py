import pandas as pd

# Load your dataset (replace 'your_file.csv' with the actual file)
df = pd.read_csv("/Users/katarinadvornak/team10/HAM10000_metadata.csv", delimiter=",")

# Check for missing values in the 'age' column
print(f"Missing values before imputation: {df['age'].isna().sum()}")

# Impute missing values using the median age per disease type
df['age'] = df.groupby('dx')['age'].transform(lambda x: x.fillna(x.median()))

# Check for missing values after imputation
print(f"Missing values after imputation: {df['age'].isna().sum()}")

# Save the cleaned dataset (optional)
df.to_csv("cleaned_data.csv", index=False)

print("Missing age values have been filled using the median per disease type.")
