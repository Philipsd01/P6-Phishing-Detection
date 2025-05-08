import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

## resolve from THIS file into an absolute folder so HF never tries the Hub
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(
    PROJECT_ROOT,
    "model",
    "roberta-base_lr6e-05_ep4_0507-1510"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

def preprocess_text(text):
    # Tokenize the text, ensuring the tensor is returned
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True)

def predict_phishing(processed_input):
    with torch.no_grad():
        outputs = model(**processed_input)
    logits = outputs.logits
    # Get the predicted label index
    predicted_label = torch.argmax(logits, dim=-1).item()
    # Assuming label 1 is phishing, label 0 is not phishing (safe)
    is_phishing = (predicted_label == 1)
    
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    if is_phishing:
        # Confidence that it IS phishing
        score = probabilities[0][1].item() 
    else:
        # Confidence that it IS NOT phishing (safe)
        score = probabilities[0][0].item()
        
    return {"is_phishing": is_phishing, "score": score}

@app.route('/predict', methods=['POST'])
def handle_prediction():
    try:
        data = request.get_json()
        if not data or 'raw_text' not in data:
            return jsonify({"error": "Missing 'raw_text' in request"}), 400

        # Use the raw text from the payload directly.
        full_email = data.get('raw_text')
        print(f"Received prediction request. Full email text: {full_email[:50]}...")  # Log for debugging

        # Preprocess the email using the tokenizer
        processed_input = preprocess_text(full_email)

        # Make prediction using the processed input and BERT model
        prediction_result = predict_phishing(processed_input)

        return jsonify({
            "is_phishing": prediction_result["is_phishing"],
            "confidence": prediction_result["score"]
        })
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Run on port 5000, accessible outside Docker