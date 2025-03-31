from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ✅ Load model and tokenizer
#model_path = "../saved_models/bert_lr2e-05_ep1_20250319-122318"  # adjust to your real path
#tokenizer = BertTokenizer.from_pretrained(model_path)
#model = BertForSequenceClassification.from_pretrained(model_path)
#model.eval()

app = FastAPI()

# ✅ Add CORS middleware (for Chrome extension)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mail.google.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define input schema
class EmailInput(BaseModel):
    body: str
    subject: str

# ✅ Prediction route using real model
@app.post("/predict")
def predict(input: EmailInput):
    print("📨 SUBJECT:", input.subject[:100])
    print("📄 BODY:", input.body[:300])  # limit just for preview

    return {
        "label": "phishing",  # dummy value
        "score": 87.5          # dummy value
    }

