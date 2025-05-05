import os
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Use a relative path so it works in both dev and Docker environments.
# Remember to change to actual model path when deploying.
model_path = os.path.join("models", "roberta-base_lr3e-06_ep3_0502-1429")

tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
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
    # Assuming label 1 is phishing
    is_phishing = (predicted_label == 1)
    # Optionally, compute a confidence score
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    score = probabilities[0][1].item()
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
            "score": prediction_result["score"]
        })
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Run on port 5000, accessible outside Docker