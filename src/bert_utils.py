from transformers import AutoTokenizer
from datasets import Dataset

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def convert_to_dataset(df):
    return Dataset.from_pandas(df)

def tokenize_dataset(dataset, tokenizer):
    # Use a local tokenization function that accesses the correct key
    def tokenize(example):
        return tokenizer(
            example['text'],  # adjust key to your text field(s)
            truncation=True,
            max_length=160,
            padding='max_length'
        )
    
    return dataset.map(tokenize, batched=True)