"""
Test script for sentiment analysis using a fine-tuned BERT model.
Written by DJ Leamen
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained('./module activities/fine_tuned_bert')  # type: ignore
tokenizer = AutoTokenizer.from_pretrained('./module activities/fine_tuned_bert')

# Test input
TEST_TEXT = 'hi there'

# Tokenize
inputs = tokenizer(TEST_TEXT, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Get prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = int(torch.argmax(predictions, dim=-1).item())

# Map back to labels
label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
confidence = predictions[0][predicted_class].item()

print(f'Text: "{TEST_TEXT}"')
print(f'Predicted sentiment: {label_map[predicted_class]}')
print(f'Confidence: {confidence:.2%}')
print('All probabilities:')
print(f'  Positive: {predictions[0][0].item():.2%}')
print(f'  Neutral: {predictions[0][1].item():.2%}')
print(f'  Negative: {predictions[0][2].item():.2%}')
