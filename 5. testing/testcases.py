"""
Practice activity: Designing test cases for ML systems
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import ipytest
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Set up the model
model = DecisionTreeClassifier()
model.fit(X, y)

def test_typical_case():
    """Test typical input case for the decision tree model."""
    input_data = np.array([[4.5, 2.3, 1.3, 0.3]])  # Example input for a flower classification model
    expected_output = 0  # Expected output for typical case (Setosa class index)
    result = model.predict(input_data)[0]
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
    
def test_edge_case_extreme_values():
    """Test edge case with extreme input values for the decision tree model."""
    input_data = np.array([[1000, 1000, 1000, 1000]])  # Extreme values for flower classification
    try:
        model.predict(input_data)
    except ValueError:
        assert True  # The model should raise a ValueError for extreme inputs
    else:
        assert False, "Expected ValueError for extreme values, but no error was raised"
        
def test_error_handling_missing_values():
    """Test error handling for missing values in the decision tree model."""
    input_data = np.array([[None, None, None, None]])  # Missing values in input
    try:
        model.predict(input_data)
    except ValueError:
        assert True  # The model should raise a ValueError for missing inputs
    else:
        assert False, "Expected ValueError for missing values, but no error was raised"