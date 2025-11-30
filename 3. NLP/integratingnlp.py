"""
Practice activity: Integrating NLP components
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""
from nltk.tokenize import word_tokenize
from transformers import pipeline

import nltk
import spacy
nltk.download('punkt_tab')

# Sample text
TEXT = "Natural Language Processing is transforming AI applications."

# Tokenize the text
tokens = word_tokenize(TEXT)
print(tokens)

nltk.download('averaged_perceptron_tagger_eng')
# Apply POS tagging
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)

# Load the pretrained model for NER
nlp = spacy.load("en_core_web_sm")

# Process the text with NER
doc = nlp(TEXT)

# Extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis') # type: ignore

# Analyze the sentiment of the text
result = sentiment_analyzer(TEXT)
print(result)
