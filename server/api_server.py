import os
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Use a relative path so it works in both dev and Docker environments.
model_path = os.path.join("models", "trained_models", "bert_phishing_20250402_123906")

tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

def preprocess_text(text):
    # Tokenize the text, ensuring the tensor is returned
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True)

def predict_phishing(processed_input):
    # Run the model inference without computing gradients
    with torch.no_grad():
        outputs = model(**processed_input)
    # Apply softmax to obtain probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Assuming your model has 2 labels: index 1 is "phishing"
    score = probabilities[0][1].item()
    # You can adjust this threshold based on your model's performance
    is_phishing = score > 0.5
    return {"is_phishing": is_phishing, "score": score}

@app.route('/predict', methods=['POST'])
def handle_prediction():
    try:
        data = request.get_json()
        if not data or 'email_body' not in data:
            return jsonify({"error": "Missing 'email_body' in request"}), 400

        email_body = data.get('email_body', '')
        email_subject = data.get('email_subject', '')

        # Combine subject and body for prediction
        full_email = email_subject + " " + email_body

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