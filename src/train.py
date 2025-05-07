import os
import argparse
from datetime import datetime
from preprocess import load_and_prepare_data
from bert_utils import get_tokenizer, convert_to_dataset, tokenize_dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    
    # Calculate standard metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    cm = confusion_matrix(labels, preds)
    # For binary classification, cm is [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    return {
        'accuracy': accuracy_score(labels, preds), 
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted'),
        'false_positive_rate': fpr
    }

    #Change learning rate and epochs here:
def train_model(model_variant, learning_rate=6e-5, epochs=4):
    # Dynamic output dir based on params + timestamp
    timestamp = datetime.now().strftime("%m%d-%H%M")
    model_name = f"{model_variant}_lr{learning_rate}_ep{epochs}_{timestamp}"
    output_dir = f"models/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    df = load_and_prepare_data('data/processed_data/combined_cleaned_sample.csv')
    dataset = convert_to_dataset(df)

    # Tokenize
    tokenizer = get_tokenizer(model_variant)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    # Save split dataset to disk
    tokenized_dataset.save_to_disk("data/tokenized_split_dataset")

    # Load model
    model = AutoModelForSequenceClassification\
                .from_pretrained(model_variant, num_labels=2)

    # Training setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        seed=42,
        num_train_epochs=epochs,
        weight_decay=0.0205,               
        eval_strategy="epoch",          
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,     
        warmup_steps=40,                
    )

    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Trainer with EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluation step on the test set
    eval_results = trainer.evaluate()
    print(f"Evaluation results for model ({model_variant}):")
    for metric, value in eval_results.items():
        print(f"{metric}: {value}")

    # Save final model and tokenizer
    trainer.model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a HF sequence-classification model")
    parser.add_argument(
        "--model_variant",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model identifier"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Define the list of models to train sequentially
    model_variants = [
        "FacebookAI/roberta-base",
        "distilbert/distilbert-base-uncased",
        "google-bert/bert-base-uncased",
        "xlnet/xlnet-base-cased",
    #    "google-bert/bert-large-uncased"   #Didn't load properly for Soya420 stuck on 1/xxx
    #    "microsoft/deberta-v3-base"        #Didn't load properly for Soya420 stuck on 1/xxx
    ]
    
    for variant in model_variants:
        print(f"Training model: {variant}")
        train_model(model_variant=variant)