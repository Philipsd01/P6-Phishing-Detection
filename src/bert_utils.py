from transformers import BertTokenizer
from datasets import Dataset

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

def convert_to_dataset(df):
    return Dataset.from_pandas(df)

def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)
    return dataset.map(tokenize_fn, batched=True)
