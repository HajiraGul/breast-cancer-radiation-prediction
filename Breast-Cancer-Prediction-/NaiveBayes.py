# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
df = pd.read_csv("breast-cancer-cleaned.csv")
print("Dataset preview:\n", df.head())

# Step 3: Separate features and target
X = df.drop("irradiat", axis=1)  # Features
y = df["irradiat"]              # Target

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Step 4: Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Initialize and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Step 8: Display results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2. Bar Plot for Precision, Recall, F1-score
metrics_df = pd.DataFrame(report).transpose().iloc[:3][['precision', 'recall', 'f1-score']]
metrics_df.plot(kind='bar', figsize=(10,6))
plt.title("Classification Metrics per Class")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.show()
