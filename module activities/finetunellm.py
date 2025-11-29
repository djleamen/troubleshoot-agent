"""
Practice activity: Fine-tuning an LLM
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import os

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

# Load a pretrained model and tokenizer (e.g., BERT for sequence classification)
MODEL_NAME = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Load the IMDb dataset
dataset: DatasetDict = load_dataset('imdb')  # type: ignore

# Convert dataset to Pandas DataFrame
df = dataset['train'].to_pandas()

# Perform train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize the dataset


def preprocess_function(examples):
    """Tokenize the input examples"""
    return tokenizer(examples['text'], padding="max_length", truncation=True)


# Convert back to Hugging Face dataset
train_data = Dataset.from_pandas(train_data)
test_data = Dataset.from_pandas(test_data)

# Apply preprocessing
train_data = train_data.map(preprocess_function, batched=True)
test_data = test_data.map(preprocess_function, batched=True)

# Import Trainer from the submodule directly

# Disable parallelism warning and MLflow logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "disable"
os.environ["HF_MLFLOW_LOGGING"] = "false"

# Ensure CPU usage if no GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a smaller, faster model like DistilBERT
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2)
model.to(device)

# Use a subset of the dataset to speed up training
train_data = train_data.select(range(1000))  # Select 1000 samples for training
# Select 200 samples for evaluation
test_data = test_data.select(range(200))

# Define a function to compute accuracy


def compute_metrics(eval_pred):
    """Compute accuracy metric for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


# Set up training arguments for faster training
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0,
    logging_steps=500,
    save_steps=1000,
    save_total_limit=1,
    gradient_accumulation_steps=1,
    fp16=False,
    report_to="none",
)

# Define the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Print evaluation results
print(f"Accuracy: {results['eval_accuracy']}")

