"""
Practice activity: Logging
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import logging

# Set up logging to a file
logging.basicConfig(filename='ml_pipeline.log', level=logging.INFO)

# Example log message
logging.info("Logging setup complete.")

import pandas as pd
from sklearn.datasets import load_breast_cancer

# Log the start of data loading
logging.info("Loading dataset...")

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
logging.info("Dataset loaded successfully.")

# Log the start of preprocessing
logging.info("Starting data preprocessing...")

# Example preprocessing: handling missing values
df.fillna(0, inplace=True)
logging.info("Missing values filled with 0.")

# Log the completion of preprocessing
logging.info("Data preprocessing completed.")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log the start of model training
logging.info("Starting model training...")

try:
    # Train the decision tree model
    model = DecisionTreeClassifier(ccp_alpha=0.0, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model trained successfully.")
except Exception as e:
    logging.error(f"Error during model training: {e}")

# Example logging of training accuracy (if applicable)
accuracy = model.score(X_train, y_train)
logging.info(f"Training accuracy: {accuracy:.2f}")

# Log the start of predictions
logging.info("Starting model predictions...")

try:
    # Make predictions
    predictions = model.predict(X_test)
    logging.info("Predictions made successfully.")
except Exception as e:
    logging.error(f"Error during predictions: {e}")

# Log the output (in production systems, limit the amount of data logged)
logging.info(f"Prediction output: {predictions[:5]}")  # Log only first 5 predictions

# Example: logging an exception during data validation
def validate_data(data):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        logging.info("Data validation successful.")
    except ValueError as e:
        logging.error(f"Data validation error: {e}")

# Validate the dataset
validate_data(df)