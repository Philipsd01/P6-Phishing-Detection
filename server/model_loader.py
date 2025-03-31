from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    return model, tokenizer
