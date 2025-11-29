"""
Practice activity: Applying LoRA
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import loralib as lora
import torch.nn as nn
from datasets import load_dataset
from transformers import (BertForSequenceClassification, BertTokenizer,
                          Trainer, TrainingArguments)

# Load a pre-trained BERT model for classification tasks
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=3)

# Print model layers to identify attention layers where LoRA can be applied
for name, module in model.named_modules():
    print(name)  # This output helps locate attention layers

# Replace linear layers in attention with LoRA layers
for name, module in model.named_modules():
    if 'attention' in name and isinstance(module, nn.Linear):
        # Replace the linear layer with a LoRA linear layer
        PARENT_NAME = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent_module = model.get_submodule(
            PARENT_NAME) if PARENT_NAME else model
        setattr(parent_module, child_name, lora.Linear(
            module.in_features, module.out_features, r=8))

# Mark only LoRA parameters as trainable
lora.mark_only_lora_as_trainable(model)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load a sample dataset (you can replace this with your own dataset)
# For this example, we'll use a sentiment analysis dataset
dataset = load_dataset("glue", "sst2")

# Tokenize the dataset
def tokenize_function(examples):
    """Tokenize the input examples using the BERT tokenizer."""
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split into train, validation, and test sets
train_data = tokenized_dataset["train"].shuffle(  # type: ignore
    seed=42).select(range(1000))  # type: ignore
val_data = tokenized_dataset["validation"].shuffle(  # type: ignore
    seed=42).select(range(200))  # type: ignore
test_data = tokenized_dataset["validation"].shuffle(  # type: ignore
    seed=42).select(range(200, 400))  # type: ignore

# Configure training parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
)

# Set up the Trainer to handle fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Begin training
trainer.train()

# Evaluate the LoRA fine-tuned model on the test set
results = trainer.evaluate(eval_dataset=test_data)  # type: ignore
print(f"Test results: {results}")

# Example: To adjust the rank in LoRA, you would modify the 'r' parameter when creating LoRA layers
# The rank was set to 8 in cell 2. To use a different rank (e.g., 2, 4, 16),
# you would need to recreate the model with the desired rank value
# in the lora.Linear() initialization

# Display the number of trainable parameters in the LoRA-adapted model
trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(
    f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
