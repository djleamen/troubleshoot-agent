"""
Practice activity: Applying QLoRA
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments
from qlora import QuantizeModel, LoRALayer, adjust_qlora_rank

# Load the pre-trained GPT-2 model
model = GPT2ForSequenceClassification.from_pretrained('gpt2')

# Quantize the model
quantized_model = QuantizeModel(model, bits=8)

# Apply LoRA to specific layers (e.g., attention layers)
for name, module in quantized_model.named_modules():
    if 'attention' in name:
        module.apply(LoRALayer)
        
# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
)

# Fine-tune the QLoRA-enhanced model
trainer = Trainer(
    model=quantized_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(eval_dataset=test_data)
print(f"Test Accuracy: {results['eval_accuracy']}")

# Adjust the rank of the low-rank matrices
adjust_qlora_rank(quantized_model, rank=4)  # Experiment with different rank values