"""
Practice activity: Applying PEFT
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

# Step 1: Prepare your data and identify the subset of parameters for fine-tuning

# Load pre-trained BERT model
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=3)

# Step 1a: Freeze all layers except the last one (classification head)
for param in model.base_model.parameters():
    param.requires_grad = False

# If you'd like to fine-tune additional layers (e.g., the last 2 layers), 
# you can unfreeze those layers as well
for layer in model.bert.encoder.layer[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# Step 2: Set up fine-tuning with PEFT

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = load_dataset("glue", "sst2")

# Tokenize the dataset


def tokenize_function(examples):
    """Tokenize the input examples"""
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_data = tokenized_dataset["train"].shuffle( # type: ignore
    seed=42).select(range(1000))  # type: ignore
val_data = tokenized_dataset["validation"].shuffle( # type: ignore
    seed=42).select(range(200))  # type: ignore
test_data = tokenized_dataset["validation"].shuffle( # type: ignore
    seed=42).select(range(200, 400))  # type: ignore

# Step 2a: Set training arguments for fine-tuning the model
training_args = TrainingArguments(
    output_dir='./results',             # Directory where results will be stored
    # Number of epochs (full passes through the dataset)
    num_train_epochs=3,
    per_device_train_batch_size=16,     # Batch size per GPU/CPU during training
    eval_strategy="epoch",        # Evaluate the model at the end of each epoch
)

# Step 2b: Fine-tune only the final classification head (since earlier layers were frozen)
trainer = Trainer(
    model=model,                        # Pre-trained BERT model with frozen layers
    args=training_args,                 # Training arguments
    train_dataset=train_data,           # Training data for fine-tuning
    # Validation data to evaluate performance during training
    eval_dataset=val_data,
)

# Step 2c: Train the model using PEFT (this performs PEFT because layers were frozen in Step 1)
trainer.train()


# Step 3: Monitor and evaluate performance
# After fine-tuning the model with PEFT, it is
# important to evaluate the model's performance
# and compare it to traditional fine-tuning methods.
# PEFT achieves similar or even better performance
# with less computational cost.

# Evaluate the model
results = trainer.evaluate(eval_dataset=test_data) # type: ignore
print(f"Test Results: {results}")


# Step 4: Optimize PEFT for your task
# PEFT can be further optimized for specific tasks
# by experimenting with different sets of parameters
# or layers to fine-tune. You can also try adjusting
# the learning rate or batch size to see how they
# impact the model’s performance.

# Example of adjusting learning rate for PEFT optimization
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=5e-5,  # Experiment with different learning rates
    num_train_epochs=5,
    per_device_train_batch_size=16,
)
