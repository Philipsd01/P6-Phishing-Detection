from preprocess import load_and_prepare_data
from bert_utils import get_tokenizer, convert_to_dataset, tokenize_dataset

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

def train_model():
    # Load and prepare data
    df = load_and_prepare_data('data/processed_data/Phishing_Email2_cleaned.csv')
    dataset = convert_to_dataset(df)

    # Tokenize
    tokenizer = get_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    # Load model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Training setup
    training_args = TrainingArguments(
        output_dir="../models/bert_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_dir="../logs"
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"]
    )

    trainer.train()

if __name__ == "__main__":
    train_model()
