from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

model = keras.models.load_model("final_model_fast.h5")

with open("class_names.json") as f:
    class_names = json.load(f)

def predict_image(img):
    img = img.resize((96, 96))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = np.argmax(pred)

    return class_names[index]

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file.stream)

    result = predict_image(img)

    return jsonify({"result": result})

app.run(host="0.0.0.0", port=10000)
