import os
import gdown

MODEL_PATH = "crop_model_final.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=131Ojm_gFajF-WEjD9fIxno8oFPGJFLbv"
    gdown.download(url, MODEL_PATH, quiet=False)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

model = tf.keras.models.load_model("crop_model_final.keras")

with open("class_names.json") as f:
    class_names = json.load(f)

def preprocess(image):
    image = image.resize((160,160))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file)

    img = preprocess(image)
    preds = model.predict(img)

    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return jsonify({
        "disease": class_names[class_id],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run()