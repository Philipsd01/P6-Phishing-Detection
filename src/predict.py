from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("../models/bert_model")
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    return pred, probs

# Example usage
email_text = "Your account has been suspended. Click here to reset your password."
prediction, confidence = predict(email_text)

print(f"Prediction: {'Phishing' if prediction == 1 else 'Legit'}")
print(f"Confidence: {confidence}")
