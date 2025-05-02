from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    result = response.json()

    # Check if the result is a list and the first element is a dictionary
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        sentiment = result[0].get('label', 'unknown')  # Safely get the 'label'
        score = result[0].get('score', 0)  # Safely get the 'score'
        return jsonify({
            "primary": sentiment.lower(),
            "score": score
        })
    else:
        return jsonify({"error": "Invalid response format"}), 500
