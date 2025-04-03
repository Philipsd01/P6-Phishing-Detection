import os
from datetime import datetime
from preprocess import load_and_prepare_data
from bert_utils import get_tokenizer, convert_to_dataset, tokenize_dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

def train_model(learning_rate=2e-5, epochs=3):
    # Dynamic output dir based on params + timestamp
    timestamp = datetime.now().strftime("%m%d-%H%M")
    model_name = f"bert_lr{learning_rate}_ep{epochs}_{timestamp}"
    output_dir = f"models/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    df = load_and_prepare_data('data/processed_data/Phishing_Email2_cleaned.csv')
    dataset = convert_to_dataset(df)

    # Print first 5 preprocessed texts
    print("Preprocessed email texts:")
    for i, example in enumerate(dataset.select(range(5))):
        # Adjust the key below to match the column name in your dataset.
        print(f"Example {i}: {example['text']}")

    # Tokenize
    tokenizer = get_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    # Save split dataset to disk
    tokenized_dataset.save_to_disk("data/tokenized_split_dataset")


    # Load model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Training setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        seed=42,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",
        save_total_limit=1
    )

    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"]
    )

    trainer.train()

    # Save final model and tokenizer
    trainer.model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_model(learning_rate=2e-5, epochs=1)
