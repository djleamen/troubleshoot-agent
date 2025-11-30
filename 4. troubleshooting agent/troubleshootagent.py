"""
Practice activity: Implementing the troubleshooting agent
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import zscore
import pandas as pd
import numpy as np

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Generate synthetic data
n_samples = 1000
data = {
    'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='h'),
    'cpu_usage': rng.normal(50, 10, n_samples),       # CPU usage in percentage
    'memory_usage': rng.normal(60, 15, n_samples),    # Memory usage in percentage
    'network_latency': rng.normal(100, 20, n_samples), # Network latency in ms
    'disk_io': rng.normal(75, 10, n_samples),         # Disk I/O in MB/s
    'error_rate': rng.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% error rate
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())


# Implement anomaly detection using Isolation Forest

def detect_anomalies(input_data):
    """Detect anomalies in the dataset using Isolation Forest."""
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(input_data)
    anomalies = model.predict(input_data)
    return anomalies


# Detect anomalies in the dataset
numeric_data = df.select_dtypes(include=[float, int])  # Only numeric columns
df['anomaly'] = detect_anomalies(numeric_data)

print(df['anomaly'].value_counts())  # -1 denotes an anomaly


# Calculate z-scores to identify anomalous values per column in anomalous rows
z_scores = numeric_data.apply(zscore)

# Function to identify anomalous columns for each row


def find_anomalous_columns(row, threshold=3):
    """Identify which columns in a row are anomalous based on z-scores."""
    return [col for col in numeric_data.columns if abs(z_scores.loc[row.name, col]) > threshold]  # type: ignore


# Apply the function to each anomalous row
df['anomalous_columns'] = df.apply(lambda row: find_anomalous_columns(
    row) if row['anomaly'] == -1 else [], axis=1)

# Display rows with anomalies and their anomalous columns
print(df[df['anomaly'] == -1][['timestamp', 'anomaly', 'anomalous_columns']])


# Train a decision tree for root cause analysis

def root_cause_analysis(X_train, y_train, X_test):
    """Train a decision tree classifier to predict root causes of anomalies."""
    model = DecisionTreeClassifier(ccp_alpha=0.0, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


# Prepare data for root cause analysis (only numeric columns)
X_TRAIN = df.drop(
    ['anomaly', 'timestamp', 'anomalous_columns'], axis=1, errors='ignore')
y_train = df['anomaly']
predicted_causes = root_cause_analysis(X_TRAIN, y_train, X_TRAIN)
print(f"Predicted causes shape: {predicted_causes.shape}")

# Example solution recommendation based on root cause


def recommend_solution(root_cause):
    """Recommend solutions based on detected root causes."""
    solutions = {
        "network_error": "Restart the network service.",
        "database_issue": "Check the database connection and restart the service.",
        "high_cpu_usage": "Optimize running processes or allocate more resources."
    }
    return solutions.get(root_cause, "No recommendation available.")


# Recommend a solution based on a detected root cause
solution = recommend_solution("network_error")
print(f"Recommended solution: {solution}")

# Simulate a network error by altering the dataset
df.loc[0, 'network_latency'] = 1000  # Simulating high network latency

# Prepare numeric data for prediction
NUMERIC_TEST_DATA = df.drop(
    ['anomaly', 'timestamp', 'anomalous_columns'], axis=1, errors='ignore')

# Run the troubleshooting agent
ANOMALIES = detect_anomalies(df.select_dtypes(include=[float, int]))
predicted_causes = root_cause_analysis(X_TRAIN, y_train, NUMERIC_TEST_DATA)
solution = recommend_solution(predicted_causes[0])
print(f"Detected issue: {predicted_causes[0]}")
print(f"Recommended solution: {solution}")
