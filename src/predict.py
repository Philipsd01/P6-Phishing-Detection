import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

class PhishingClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_len = 256

    def predict(self, email_data):
        # Process new data
        if isinstance(email_data, dict):  # Single email
            texts = [email_data['subject'] + ' ' + email_data['body']]
        else:  # DataFrame
            texts = (email_data['subject'] + ' ' + email_data['body']).tolist()

        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Predict
        self.model.eval()
        with torch.no_grad():
            inputs = {
                'input_ids': encodings['input_ids'].to(self.device),
                'attention_mask': encodings['attention_mask'].to(self.device)
            }
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return preds.cpu().numpy(), probs.cpu().numpy()

def predict_new_data():
    # Load model
    classifier = PhishingClassifier('../models/trained_models/bert_phishing')

    # Example usage
    new_emails = pd.read_csv('processed_data/Nigerian_Fraud_cleaned.csv')  # Your prediction data
    predictions, probabilities = classifier.predict(new_emails)
    
    # Add predictions to dataframe
    new_emails['prediction'] = predictions
    new_emails['probability'] = probabilities[:, 1]  # Probability of being phishing
    
    # Save results
    new_emails.to_csv('../data/predictions.csv', index=False)

if __name__ == '__main__':
    predict_new_data()