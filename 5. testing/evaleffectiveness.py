"""
Practice activity: Evaluating agent effectiveness
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

from sklearn.metrics import accuracy_score, precision_score
import numpy as np

# Make predictions on the test set
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate precision (average='weighted' to handle multiple classes)
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

import time

# Measure response time for multiple iterations
start_time = time.time()
for _ in range(25):
    model.predict(x_test)
end_time = time.time()

average_response_time = (end_time - start_time) / 25
print(f"Average Response Time: {average_response_time:.4f} seconds")

import psutil

# Monitor resource usage
cpu_usage = psutil.cpu_percent()
memory_usage = psutil.virtual_memory().percent

# for better results, measure CPU usage while inference is active,
# and measure memory usage against a baseline before the model is loaded
print(f"CPU Usage: {cpu_usage}%")
print(f"Memory Usage: {memory_usage}%")

import numpy as np
import time

# Ensure correct shape before repeating
print("Original x_test shape:", x_test.shape)  # Expected: (10000, 28, 28)

# Properly duplicate test data along batch axis
large_input = np.repeat(x_test, 10, axis=0)  # Expands batch size only

# Verify new shape
print("Large input shape after fix:", large_input.shape)  # Should be (100000, 28, 28)

# Measure performance under stress
start_time = time.time()
model.predict(large_input)  # Now matches model input (batch_size, 28, 28)
end_time = time.time()

print(f"Response Time under Stress (Reduced Size): {end_time - start_time:.4f} seconds")

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Example data generation for demonstration (replace with actual data)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
agent_model = RandomForestClassifier()  # Replace with your actual model

# Perform 5-fold cross-validation
cv_scores = cross_val_score(agent_model, X, y, cv=5)

# Print the cross-validation scores for each fold
print(f'Cross-Validation Scores: {cv_scores}')

# Print the mean and standard deviation of the scores
print(f'Mean CV Score: {cv_scores.mean():.4f}')
print(f'Standard Deviation of CV Scores: {cv_scores.std():.4f}')