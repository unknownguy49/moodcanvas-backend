from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

API_URL = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Text is required"}), 400

        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            return jsonify({"error": result["error"]}), 500

        top = sorted(result[0], key=lambda x: x['score'], reverse=True)[0]
        return jsonify({
            "primary": top['label'].lower(),
            "score": top['score']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
