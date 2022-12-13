import json
import replicate
from flask import (
    Flask,
    jsonify,
    render_template,
    send_from_directory,
    request,
)
import random 
import os

app = Flask(__name__)

# Render index page
@app.route("/")
def index():
    return render_template("index.html")


# Predict
@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json()
    prompt = body['prompt']

    # Get model
    model = replicate.models.get("prompthero/openjourney")
    version = model.versions.get(
        "9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb"
    )

    output = version.predict(prompt=prompt)

    return jsonify({"prediction_url": output[0]})

if __name__ == "__main__":
    app.run(debug=True)
