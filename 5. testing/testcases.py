"""
Practice activity: Designing test cases for ML systems
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Set up the model
model = DecisionTreeClassifier(ccp_alpha=0.0, random_state=42)
model.fit(X, y)


def test_typical_case():
    """Test typical input case for the decision tree model."""
    input_data = np.array([[4.5, 2.3, 1.3, 0.3]]
                          )  # Example input for a flower classification model
    # Expected output for typical case (Setosa class index)
    expected_output = 0
    result = model.predict(input_data)[0]
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_edge_case_extreme_values():
    """Test edge case with extreme input values for the decision tree model."""
    input_data = np.array([[1000, 1000, 1000, 1000]]
                          )  # Extreme values for flower classification
    # Decision trees can handle extreme values, so we just verify it produces a valid prediction
    result = model.predict(input_data)
    assert result[0] in [
        0, 1, 2], f"Expected a valid class (0, 1, or 2), but got {result[0]}"


def test_error_handling_missing_values():
    """Test error handling for missing values in the decision tree model."""
    input_data = np.array([[None, None, None, None]]
                          )  # Missing values in input
    with pytest.raises((ValueError, TypeError)):
        model.predict(input_data)
