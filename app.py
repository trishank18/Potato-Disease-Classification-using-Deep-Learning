import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMAGE_SIZE = (256, 256)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
model = load_model(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    preview = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            error = "Choose an image to classify."
        elif "." not in uploaded.filename or uploaded.filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
            error = "Upload a JPG or PNG image."
        else:
            try:
                image = Image.open(BytesIO(uploaded.read())).convert("RGB")
                preview_buffer = BytesIO()
                preview_image = image.copy()
                preview_image.thumbnail((600, 600))
                preview_image.save(preview_buffer, format="JPEG")
                preview = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.getvalue()).decode("ascii")

                model_image = image.resize(IMAGE_SIZE)
                image_array = np.expand_dims(np.asarray(model_image, dtype=np.float32), axis=0)
                probabilities = model.predict(image_array, verbose=0)[0]
                class_index = int(np.argmax(probabilities))
                result = {
                    "label": CLASS_NAMES[class_index],
                    "confidence": round(float(probabilities[class_index]) * 100, 2),
                }
            except (UnidentifiedImageError, OSError, ValueError):
                error = "That file could not be read as an image. Try another JPG or PNG."

    return render_template("index.html", result=result, error=error, preview=preview)


if __name__ == "__main__":
    app.run(debug=True)
