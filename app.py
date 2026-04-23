from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

model = None   # 🔥 initially empty

with open("class_names.json") as f:
    class_names = json.load(f)


# ==============================
# 🔹 Load model ONLY when needed
# ==============================
def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = keras.models.load_model("final_model_fast.h5")
    return model


# ==============================
# 🔹 Prediction function
# ==============================
def predict_image(img):
    model = get_model()

    img = img.resize((96, 96))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return class_names[index], confidence


# ==============================
# 🔹 Routes
# ==============================

@app.route("/")
def home():
    return "API running ✅"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    label, confidence = predict_image(img)

    return jsonify({
        "result": label,
        "confidence": confidence
    })
