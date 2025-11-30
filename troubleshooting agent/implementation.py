"""
Practice activity: Coding in Python
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import nltk
import spacy
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Sample user query
QUERY = "My laptop is overheating after the update."

# Tokenize the query
tokens = word_tokenize(QUERY)
print(tokens)

# Apply POS tagging to the tokens
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)

# Load the pre-trained model for NER
nlp = spacy.load("en_core_web_sm")

# Apply NER to the query
doc = nlp(QUERY)

# Extract and print entities
for ent in doc.ents:
    print(ent.text, ent.label_)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis') # type: ignore

# Analyze the sentiment of the query
result = sentiment_analyzer(QUERY)
print(result)

# Sample knowledge base
knowledge_base = {
    "overheating": "Check your cooling system, clean the fans, and ensure proper ventilation.",
    "slow performance": "Close unnecessary applications, restart your system, and check for malware."
}

# Function to retrieve solutions
def get_solution(issue):
    return knowledge_base.get(issue, "No solution found for this issue.")

# Example usage
print(get_solution("overheating"))

def troubleshoot(query):
    if "overheating" in query.lower():
        return get_solution("overheating")
    elif "slow" in query.lower():
        return get_solution("slow performance")
    else:
        return "Can you provide more details about the issue?"

# Example usage
response = troubleshoot(QUERY)
print(response)
