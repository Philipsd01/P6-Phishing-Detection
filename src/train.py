import os
from datetime import datetime
from preprocess import load_and_prepare_data
from bert_utils import get_tokenizer, convert_to_dataset, tokenize_dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback


def train_model(learning_rate=1.5e-5, epochs=4, model_variant="distilbert-base-uncased"):
    # Dynamic output dir based on params + timestamp
    timestamp = datetime.now().strftime("%m%d-%H%M")
    model_name = f"{model_variant}_lr{learning_rate}_ep{epochs}_{timestamp}"
    output_dir = f"models/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    df = load_and_prepare_data('data/processed_data/combined_cleaned_sample.csv')
    dataset = convert_to_dataset(df)

    # Tokenize
    tokenizer = get_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    # Save split dataset to disk
    tokenized_dataset.save_to_disk("data/tokenized_split_dataset")

    # Load model
    model = BertForSequenceClassification.from_pretrained(model_variant, num_labels=2)

    # Training setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        seed=42,
        num_train_epochs=epochs,
        weight_decay=0.05,               # Increase weight decay for extra regularization
        evaluation_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,     # Enable best model loading
        warmup_steps=500,                # Warmup steps before reaching the set learning rate
    )

    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Trainer with EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # Save final model and tokenizer
    trainer.model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_model(learning_rate=1.5e-5, epochs=4, model_variant="distilbert-base-uncased")
