"""
Practice activity: Implementing mechanisms
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Explore the dataset
print(df.info())
print(df.head())

def validate_data(data):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if data.isnull().values.any():
            raise ValueError("Missing values detected in the dataset.")
        print("Data validation successful.")
    except ValueError as e:
        print(f"Data validation error: {e}")

# Validate the dataset
validate_data(df)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement a decision tree model with error handling
def train_model(X_train, y_train):
    try:
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        print("Model trained successfully.")
        return model
    except ValueError as e:
        print(f"Model training error: {e}")
        return None

# Train the model
model = train_model(X_train, y_train)

import logging

# Set up logging to a file
logging.basicConfig(filename='ml_errors.log', level=logging.ERROR)

def validate_data_with_logging(data):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if data.isnull().values.any():
            raise ValueError("Missing values detected in the dataset.")
        print("Data validation successful.")
    except ValueError as e:
        logging.error(f"Data validation error: {e}")

# Validate the dataset and log errors
validate_data_with_logging(df)

# Introduce missing values to test error handling
df_with_missing = df.copy()
df_with_missing.iloc[0, 0] = None

# Validate the modified dataset
validate_data_with_logging(df_with_missing)