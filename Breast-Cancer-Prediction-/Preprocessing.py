import pandas as pd

# Load the dataset from a local CSV file
column_names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
df = pd.read_csv('breast-cancer.data', header=None, names=column_names)

# Display first few rows
print(df.head())

# Get dataset structure
print(df.info())

# Summary statistics
print(df.describe(include='all'))

# Check for missing values
print(df.isnull().sum())

# Fill missing numeric values with column mean (example only)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.isnull().sum()

# Count duplicates
print(df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()

print(df.duplicated().sum())

print("Inconsistencies of Irradiat")
print(df['irradiat'].unique())
# Normalize the species column
df['irradiat'] = df['irradiat'].str.lower().str.strip()
# Check unique values
print(df['irradiat'].unique())

print("Inconsistencies of Class")
print(df['class'].unique())
# Normalize the species column
df['class'] = df['class'].str.lower().str.strip()
# Check unique values
print(df['class'].unique())

print("Inconsistencies of menopause")
print(df['menopause'].unique())


df.to_csv('breast-cancer-cleaned.csv', index=False)
print("Cleaned dataset saved as 'breast-cancer-cleaned.csv'")