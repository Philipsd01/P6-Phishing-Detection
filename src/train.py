import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

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
        'data/processed_data/Nazario_cleaned.csv'
    #    'data/processed_data/CEAS_08_cleaned.csv'
    #    'data/processed_data/Enron_cleaned.csv'
    #    'data/processed_data/Ling_cleaned.csv'

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
    EPOCHS = 2
    MODEL_NAME = 'bert-base-uncased'

    # Load data
    data = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        data['text'], data['label'], test_size=0.4, random_state=42
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
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("GPU not available")
        exit()  # Exit the program if GPU is not available
    device = torch.device('cuda')
    print(f"Using device: {device}")
    # Save split dataset to disk
    tokenized_dataset.save_to_disk("data/tokenized_split_dataset")


    # Load model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Training setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        seed=42,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",
        save_total_limit=1
    )

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Move data to GPU
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

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
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('models', 'trained_models', f'bert_phishing_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model successfully saved to {save_dir}")
    except Exception as e:
        print(f"Failed to save model: {e}")

if __name__ == '__main__':
    train()