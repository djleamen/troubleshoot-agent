"""
Practice Activity: Applying evaluation metrics in fine-tuning models
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

# Example 1: Accuracy

# When to use: Accuracy is the most intuitive metric and
# works well for balanced datasets where the number of
# positive and negative instances is approximately equal.
# For instance, if you are classifying images of cats and
# dogs and both classes are equally represented in your dataset,
# accuracy provides a clear indication of how well the model
# is performing overall.

import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

# Actual labels and predicted labels from your model
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")


# Example 2: Precision

# When to use: Precision is especially useful when
# false positives carry a significant cost. For example,
# in fraud detection, high precision ensures that when
# the model flags a transaction as fraudulent, it is
# likely to be correct, reducing the cost of investigating
# legitimate transactions.


# Calculate precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")


# Example 3: Recall (Sensitivity)

# When to use: Recall is important when false negatives are
# costly. In medical diagnoses, for example, recall ensures
# that a model identifies most of the actual positive cases
# (e.g., patients with a disease), even if that means some
# false positives occur.


# Calculate recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")


# Example 4: F1 score

# When to use: The F1 score is a balance between precision
# and recall. It is particularly useful in cases of imbalanced
# datasets, in which one class is far more prevalent than the
# other. For example, in fraud detection, you want a balance
# between precision (reducing false positives) and recall
# (catching as many fraud cases as possible).


# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")


# Example 5: Confusion matrix

# When to use: A confusion matrix is useful when you want to
# visualize how well the model is performing on both classes.
# It shows you the number of true positives (TPs), true negatives
# (TNs), false positives (FPs), and false negatives (FNs),
# helping you understand where the model is making errors.


# Generate confusion matrix
matrix = confusion_matrix(y_true, y_pred)
print(matrix)


# Example 6: ROC-AUC

# When to use: ROC-AUC is ideal for binary classification tasks
# and helps evaluate how well your model distinguishes between the
# positive and negative classes. This is particularly useful in
# imbalanced datasets, in which accuracy may not tell the full story.


# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC-AUC: {roc_auc}")


# Example 7: Loss function

# When to use: Loss is used primarily during model training
# to show how well the model is learning. For example, in
# classification problems, cross-entropy loss measures how
# far off the model’s predictions are from the actual values.

# A lower loss typically indicates a better model during training.


# Define the cross-entropy loss function
# CrossEntropyLoss is used for classification tasks where the
# model outputs class probabilities.

# It combines LogSoftmax and Negative Log Likelihood Loss into
# one function, making it efficient for such tasks.
loss_fn = nn.CrossEntropyLoss()

# Example prediction and actual class (as tensors)
# Here, we create a tensor called 'output' representing the
# predicted scores (unnormalized) for two data points.

# Each row corresponds to a data point, and the values
# represent the scores for each class.

# Note that CrossEntropyLoss internally applies the softmax
# function to these scores to obtain probabilities.
output = torch.tensor([[0.5, 1.5], [2.0, 0.5]])

# 'target' is a tensor representing the actual classes for the two data points.
# In this example, the first data point belongs to class 1, and the
# second data point belongs to class 0.

# These class indices are zero-based, meaning 0 represents the first class,
# 1 represents the second, and so on.
target = torch.tensor([1, 0])

# Calculate loss
# The CrossEntropyLoss function will take the predicted
# scores ('output') and the actual labels ('target')

# to compute the loss value, which quantifies how well the
# model's predictions match the actual labels.

# Lower loss values indicate better predictions, while higher
# values indicate more errors.
loss = loss_fn(output, target)

# Print the computed loss value
# '.item()' is used to get the Python scalar value from the tensor containing the loss.
print(f"Loss: {loss.item()}")
