import os
from flask import Flask, jsonify
from flask import request
from inference import get_prediction
from const import classes
from flask_cors import CORS  # Import Flask-CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        result = get_prediction(image_bytes=img_bytes)
        print(result)
        return jsonify({'class_id': result.item(), 'class_name': classes[result.item()]})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))