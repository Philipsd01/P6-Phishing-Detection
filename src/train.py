import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    # Load cleaned datasets
    files = [
        'data/processed_data/CEAS_08_cleaned.csv'
    #    'data/processed_data/Enron_cleaned.csv',
    #    'data/processed_data/Ling_cleaned.csv',

    # Add other cleaned files for more training data
    ]
    
    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)
    
    # Combine subject and body since bert expects a single text input
    data['text'] = data['subject'] + ' ' + data['body']
    return data

def train():
    # Config
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 1
    MODEL_NAME = 'bert-base-uncased'

    # Load data
    data = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    ignore_mismatched_sizes=True
)

    # Create dataloaders
    train_dataset = EmailDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
    val_dataset = EmailDataset(X_val.tolist(), y_val.tolist(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{EPOCHS} \n"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()

        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {correct/len(val_dataset):.4f}')

    # Save model
    model.save_pretrained('../models/trained_models/bert_phishing')
    tokenizer.save_pretrained('../models/trained_models/bert_phishing')

if __name__ == '__main__':
    train()