"""
Practice activity: Implementing NLP for troubleshooting
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline

import spacy

# Sample troubleshooting query
text = "My laptop is overheating after the update."

# Tokenize the text
tokens = word_tokenize(text)
print(tokens)

# Apply POS tagging to the tokens
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)

# Load the pretrained NER model
nlp = spacy.load("en_core_web_sm")

# Process the text with NER
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis') # type: ignore

# Analyze the sentiment of the troubleshooting query
result = sentiment_analyzer(text)
print(result)
