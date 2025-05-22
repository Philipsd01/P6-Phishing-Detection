import os
import csv
import re
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, default_data_collator
from bert_utils import get_tokenizer, tokenize_dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datasets import load_dataset
from torch.utils.data import DataLoader

def validate_model(model, data_path, model_name):
    tokenizer = get_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load the already cleaned CSV file from the validation folder.
    ds = load_dataset("csv", data_files={"validation": data_path})

    # Combine "subject" and "body" into a single "text" column.
    def combine_fields(example):
        subject = example["subject"] if example["subject"] is not None else ""
        body = example["body"] if example["body"] is not None else ""
        example["text"] = (subject + " " + body).strip()
        return example

    ds["validation"] = ds["validation"].map(combine_fields)
    # Remove the now redundant "subject" and "body" columns.
    ds["validation"] = ds["validation"].remove_columns(["subject", "body"])
    ds["validation"] = ds["validation"].filter(lambda ex: ex["label"] is not None)

    validation_set = tokenize_dataset(ds["validation"], tokenizer)
    validation_set.set_format("torch")

    dataloader = DataLoader(validation_set, batch_size=32, collate_fn=default_data_collator)
    all_preds, all_labels, all_confidences = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            # Compute probabilities using softmax.
            probs = F.softmax(outputs.logits, dim=-1)
            # Get predicted class and its confidence
            confidences, preds = torch.max(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())



    # Compute classification metrics
    metrics = classification_report(
        all_labels, all_preds, labels=[0, 1], digits=4, output_dict=True
    )
    print("Classification Metrics:")
    print("{:12} {:>10} {:>10}".format("", "0", "1"))
    print("{:12} {:>10.4f} {:>10.5f}".format(
          "Precision:", 
          metrics["0"]["precision"], metrics["1"]["precision"]))
    print("{:12} {:>10.4f} {:>10.5f}".format(
          "Recall:", 
          metrics["0"]["recall"], metrics["1"]["recall"]))
    print("{:12} {:>10.4f} {:>10.5f}".format(
          "F1-score:", 
          metrics["0"]["f1-score"], metrics["1"]["f1-score"]))

    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.5f}")

    # Calculate and print False Positive Rate (FPR) for binary classification.
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"False Positive Rate: {fpr:.5f}")
    else:
        print("Confusion matrix is not binary, unable to compute False Positive Rate.")

    # Compute and print the average confidence score.
    avg_confidence = np.mean(all_confidences)
    print(f"Average Confidence Score: {avg_confidence:.4f}")

def get_learning_rate_from_model_path(model_path):
    # Extracts the learning rate string from model_path using regex.
    pattern = r"lr([\d\.e-]+)"
    match = re.search(pattern, model_path.lower())
    if match:
        return match.group(1)
    return "N/A"

if __name__ == "__main__":
    # Use the already cleaned CSV file from the cleaned folder
    data_path = "data/validation/cleaned/combined_cleaned_sample.csv"

    # List of models to validate with their paths and model names.
    models_info = [
        {
            "model_path": "models/distilbert/distilbert-base-uncased_lr6e-05_ep4_0514-1628",
            "model_name": "distilbert-base-uncased",
        },
        {
            "model_path": "models/FacebookAI/roberta-base_lr6e-05_ep4_0514-1622",
            "model_name": "roberta-base",
        },
        {
            "model_path": "models/xlnet/xlnet-base-cased_lr6e-05_ep4_0514-1632",
            "model_name": "xlnet-base-cased",
        },
        {
            "model_path": "models/google-bert/bert-base-uncased_lr6e-05_ep4_0514-1641",
            "model_name": "bert-base-uncased",
        }
        
    ]

    for info in models_info:
        model_path = info["model_path"]
        model_name = info["model_name"]
        lr = get_learning_rate_from_model_path(model_path)
        print(f"\nValidating Model: {model_name} | Learning Rate: {lr}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        )
        validate_model(
            model=model,
            data_path=data_path,
            model_name=model_name
        )