from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS to allow cross-origin requests
CORS(app)

# Load the sentiment analysis model from Hugging Face
model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Endpoint for analyzing sentiment
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get data from the POST request
        data = request.get_json()

        # Extract text from the request
        text = data.get("text", "")

        # Check if the text is provided
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Analyze sentiment using the model
        result = model(text)[0]

        # Return the result as JSON
        return jsonify({
            "primary": result['label'].lower(),
            "score": result['score']
        })


    except Exception as e:
        # In case of an error, return a 500 error with the error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
