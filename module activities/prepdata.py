"""
Practice activity: Preparing a dataset for fine-tuning 
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

# Step 1: Import data set

import random  # Random module for generating random numbers and selections
import re  # Import the `re` module for working with regular expressions

import pandas as pd
import torch  # Import PyTorch library
from nltk.corpus import wordnet  # NLTK's WordNet corpus for finding synonyms
# Import necessary libraries
# Import function to split dataset
from sklearn.model_selection import train_test_split
# Import modules to create datasets and data loaders
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

# Load dataset
# from https://huggingface.co/datasets/stepp1/tweet_emotion_intensity/tree/main
data = pd.read_csv("hf://datasets/stepp1/tweet_emotion_intensity/train.csv")

# Preview the data
print(data.head())


# Step 2: Clean the text
# This step is cleaning the raw text data to remove
# unnecessary characters, such as URLs, special symbols,
# or HTML tags, and to normalize the text by converting it to lowercase.
#
# Make a new column called cleanedText that is equal to
# the data in the Tweet column that has had this cleanedText function applied to it.


# Function to clean the text

def clean_text(text):
    """Clean the input text and convert to lowercase."""
    text = text.lower()  # Convert all text to lowercase for uniformity
    text = re.sub(r'http\S+', '', text)  # Remove URLs from the text
    text = re.sub(r'<.*?>', '', text)  # Remove any HTML tags from the text
    # Remove punctuation, keep only words and spaces
    text = re.sub(r'[^\w\s]', '', text)
    return text  # Return the cleaned text


# Assume `data` is a pandas DataFrame with a column named 'text'
# Apply the cleaning function to each row of the 'text' column
data['cleaned_text'] = data['tweet'].apply(clean_text)

# Print the first 5 rows of the cleaned text to verify the cleaning process
print(data['cleaned_text'].head())


# Step 3: Handle missing data
# We now handle missing or incomplete data in your dataset.
# You can either remove rows with missing data or fill them
# with placeholders, ensuring the dataset is complete for training.

# Check for missing values in the dataset
print(data.isnull().sum())  # Print the count of missing values for each column

# Option 1: Remove rows with missing data in the 'cleaned_text' column
# Drop rows where 'cleaned_text' is NaN (missing)
data = data.dropna(subset=['cleaned_text'])

# Option 2: Fill missing values in 'cleaned_text' with a placeholder
# Replace NaN values in 'cleaned_text' with 'unknown'
data['cleaned_text'].fillna('unknown', inplace=True)


# Step 4: Tokenization
# After cleaning the text, we tokenize it.
# Tokenization splits the text into individual
# words or subwords that can be used by the model.
# We will use the BERT tokenizer to ensure compatibility
# with the Brie-trained model you are fine-tuning.


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the cleaned text
tokens = tokenizer(
    data['cleaned_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt'
)

print(tokens['input_ids'][:5])  # Preview the first 5 tokenized examples

# Import necessary modules

# Define a function to find and replace a word with a synonym


def synonym_replacement(word):
    """Get all synsets (sets of synonyms) for the given word from WordNet"""
    synonyms = wordnet.synsets(word)

# If the word has synonyms, randomly choose one synonym, otherwise return the original word
    if synonyms:
        # Select a random synonym and get the first lemma (word form) of that synonym
        synset = random.choice(synonyms)
        if synset and synset.lemmas():
            return synset.lemmas()[0].name()

# If no synonyms are found, return the original word
    return word

# Define a function to augment text by replacing words with synonyms randomly


def augment_text(text):
    """Split the input text into individual words"""
    words = text.split()  # Split the input text into individual words

# Replace each word with a synonym with a probability of 20% (random.random() > 0.8)
    augmented_words = [
        synonym_replacement(word) if random.random() > 0.8 else word
        # If random condition met, replace
        for word in words]  # Iterate over each word in the original text

# Join the augmented words back into a single string and return it
    return ' '.join(augmented_words)


# Apply the text augmentation function to the 'cleaned_text' column in a DataFrame
# Create a new column 'augmented_text' containing the augmented version of 'cleaned_text'
data['augmented_text'] = data['cleaned_text'].apply(augment_text)


# Step 5: Structure the data for fine-tuning
# You can fine-tune your model once the dataset is
# cleaned and tokenized. The next step is structuring
# the data for fine-tuning.
#
# Import Torch, TensorDataset and DataLoader.
# We will convert the tokens into PyTorch tensors.
# We will define a mapping function that sets the tweet
# sentiment intensity from high to 1, from medium to 0.5,
# and from low to 0. Then, we will apply that function to
# each item in sentiment_intensity, and then we will drop
# any rows where sentiment_intensity is none, where
# sentiment_intensity was something other than high, medium,
# or low. Finally, we will convert the sentiment_intensity
# column to a tensor.


# Convert tokenized data into PyTorch tensors
input_ids = tokens['input_ids']  # Extract input IDs from the tokenized data
# Extract attention masks from the tokenized data
attention_masks = tokens['attention_mask']

# Define a mapping function


def map_sentiment(value):
    """Map sentiment intensity values to numerical labels"""
    if value == "high":
        return 1
    elif value == "medium":
        return 0.5
    elif value == "low":
        return 0
    else:
        return None  # Handle unexpected values, if any


# Apply the function to each item in 'sentiment_intensity'
data['sentiment_intensity'] = data['sentiment_intensity'].apply(map_sentiment)

# Drop any rows where 'sentiment_intensity' is None
data = data.dropna(subset=['sentiment_intensity']).reset_index(drop=True)

# Convert the 'sentiment_intensity' column to a tensor
labels = torch.tensor(data['sentiment_intensity'].tolist())


# Step 6: Split the Dataset
# Finally, we split the dataset into training,
# validation, and test sets. This ensures that
# your model is trained on one portion of the
# data while its performance is monitored and
# tested on unseen examples. This includes
# organizing the tokenized data into PyTorch
# TensorDataset objects, ready for training.


# First split: 15% for test set, the rest for training/validation
train_val_inputs, test_inputs, train_val_masks, test_masks, train_val_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.15, random_state=42
)

# Second split: 20% for validation set from remaining data
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    train_val_inputs, train_val_masks, train_val_labels, test_size=0.2, random_state=42
)

# Create TensorDataset objects for each set, including attention masks
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# Create DataLoader objects
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print("Training, validation, and test sets are prepared with attention masks!")
