from transformers import BertForSequenceClassification
from bert_utils import get_tokenizer, convert_to_dataset, tokenize_dataset
from preprocess import load_and_prepare_data
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_from_disk

import torch

def evaluate_model(model, data_path):
    tokenizer = get_tokenizer()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Load pre-split test set from disk
    dataset = load_from_disk("data/tokenized_split_dataset")
    test_set = dataset["test"]
    test_set.set_format("torch")



    dataloader = DataLoader(test_set, batch_size=16, collate_fn=default_data_collator)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            labels = batch["labels"].to(model.device)

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))
    print("Accuracy:", accuracy_score(all_labels, all_preds))


if __name__ == "__main__":
    model_path = "./saved_models/bert_lr2e-05_ep1_20250319-113429"

    model = BertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )

    evaluate_model(
        model=model,
        data_path="data/processed_data/Phishing_Email2_cleaned.csv"
    )
