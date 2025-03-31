from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model_path = "../saved_models/bert_lr2e-05_ep1_20250319-113429"

tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()


app = FastAPI()

# ✅ Add CORS middleware (for Chrome extension)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["https://mail.google.com"] for more strict security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define input schema
class EmailInput(BaseModel):
    subject: str
    body: str

@app.post("/predict")
def predict(input: EmailInput):
    # Combine subject and body
    full_text = input.subject + " " + input.body

    # DEBUG: Print the actual input text
    print("🧪 Full Input Text:", repr(full_text))

    # Tokenize
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

        # DEBUG: Logits before softmax
        print("📊 Logits:", outputs.logits)

        # Apply softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # DEBUG: Show both class probabilities
        print("🔍 Probabilities (Legit, Phishing):", probs)

        phishing_score = probs[0][1].item()

    return {
        "label": "phishing" if phishing_score > 0.5 else "legit",
        "score": round(phishing_score * 100, 2)
    }



