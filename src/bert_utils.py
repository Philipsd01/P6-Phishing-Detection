from transformers import BertTokenizer
from datasets import Dataset

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

def convert_to_dataset(df):
    return Dataset.from_pandas(df)

def tokenize_dataset(dataset, tokenizer):
    # Use a local tokenization function that accesses the correct key
    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)
    return dataset.map(tokenize_fn, batched=True)

# Alternatively, if you want to keep a standalone tokenize_fn
# def tokenize_fn(example):
#     # Make sure 'tokenizer' is defined or passed in.
#     # And note that the key must match the column name in your dataset (i.e. "text")
#     return tokenizer(example["text"], padding="max_length", truncation=True)
