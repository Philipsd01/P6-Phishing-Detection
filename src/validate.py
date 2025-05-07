import os
import csv
import torch
from transformers import AutoModelForSequenceClassification, default_data_collator
from bert_utils import get_tokenizer, tokenize_dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datasets import load_dataset
from torch.utils.data import DataLoader

def evaluate_model(model, data_path, model_name):
    tokenizer = get_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load the already cleaned CSV file from the validation folder
    ds = load_dataset("csv", data_files={"validation": data_path})

    # Combine "subject" and "body" into a single "text" column
    def combine_fields(example):
        example["text"] = example["subject"] + " " + example["body"]
        return example

    ds["validation"] = ds["validation"].map(combine_fields)
    # Remove the now redundant "subject" and "body" columns
    ds["validation"] = ds["validation"].remove_columns(["subject", "body"])

    ds["validation"] = ds["validation"].filter(lambda ex: ex["label"] is not None)

    validation_set = tokenize_dataset(ds["validation"], tokenizer)
    validation_set.set_format("torch")

    dataloader = DataLoader(validation_set, batch_size=32, collate_fn=default_data_collator)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute and display metrics
    report = classification_report(all_labels, all_preds, digits=4)
    print(f"Classification Report for {model_name}:\n", report)

    acc = accuracy_score(all_labels, all_preds)
    print("Accuracy:", acc)

    # Calculate False Positive Rate (FPR) for binary classification:
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"False Positive Rate:, {fpr:.5f}\n")
    else:
        print("Confusion matrix is not binary, unable to compute False Positive Rate.")

if __name__ == "__main__":
    # Use the already cleaned CSV file from the cleaned folder
    data_path = "data/validation/cleaned/combined_cleaned.csv"



#    model_path = "models/FacebookAI/roberta-base_lr6e-05_ep4_0507-1510"
#    model_path = "models/distilbert/distilbert-base-uncased_lr6e-05_ep4_0507-1516"
#    model_path = "models/google-bert/bert-base-uncased_lr6e-05_ep4_0507-1519"
    model_path = "models/xlnet/xlnet-base-cased_lr6e-05_ep4_0507-1525"


#    model_name = "roberta-base" 
#    model_name = "distilbert-base-uncased"
#    model_name = "bert-base-uncased"
    model_name = "xlnet-base-cased"


    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )

    evaluate_model(
        model=model,
        data_path=data_path,
        model_name=model_name
    )