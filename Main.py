import pandas as pd

# Open the file and read the content
with open("HAM10000_metadata", "r", encoding="utf-8") as file:
    content = file.read()
    print(content)  # Prints the whole file

# You can also load the file into a pandas DataFrame (assuming it's a CSV or tabular data)
df = pd.read_csv("HAM10000_metadata")  # This assumes it's a CSV file
print(df.head())  # Prints the first 5 rows of the DataFrame
