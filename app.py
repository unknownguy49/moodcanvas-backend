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
    # Get the JSON data from the request
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Send POST request to Hugging Face sentiment analysis model API
    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    # Parse the response JSON and handle the results
    result = response.json()

    if isinstance(result, list) and len(result) > 0:
        sentiment = result[0].get('label', 'unknown')  # Check if 'label' exists
        score = result[0].get('score', 0.0)  # Check if 'score' exists
    else:
        return jsonify({"error": "Invalid response from model"}), 500

    # Returning result in the original format
    return jsonify({
        "primary": sentiment.lower(),
        "score": score
    })

if __name__ == '__main__':
    app.run(debug=True)
