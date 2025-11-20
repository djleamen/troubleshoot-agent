"""
Practice activity: Model and dataset selection
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

from transformers import BertTokenizer, BertForSequenceClassification
from nlpaug.augmenter.word import BackTranslationAug
from datasets import load_dataset, Dataset

if __name__ == '__main__':
    # Load pre-trained BERT model and tokenizer for classification
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=3)

    # Model and tokenizer are now ready for fine-tuning

    # Initialize the backtranslation augmenter (English -> French -> English)
    back_translation_aug = BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')

    # Example text to augment
    TEXT = "The weather is great today."

    # Perform backtranslation to create augmented text
    augmented_text = back_translation_aug.augment(TEXT)

    print("Original text:", TEXT)
    print("Augmented text:", augmented_text)

    # Load the IMDB movie reviews dataset for sentiment analysis
    dataset = load_dataset('imdb')

    # Split the dataset into training and validation sets using HuggingFace's split method
    train_dataset: Dataset = dataset['train']  # type: ignore
    train_test_split_dataset = train_dataset.train_test_split(test_size=0.2)
    train_data = train_test_split_dataset['train']
    val_data = train_test_split_dataset['test']


    def tokenize_function(examples):
        """Convert the data into the format required for tokenization"""
        return tokenizer(examples['text'], padding='max_length', truncation=True)


    # Tokenize the dataset using the map function
    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_val = val_data.map(tokenize_function, batched=True)

    print(f"Training samples: {len(tokenized_train)}")
    print(f"Validation samples: {len(tokenized_val)}")
    print("Dataset preparation complete!")
