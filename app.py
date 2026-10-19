"""Flask web application for the three-class potato disease model."""
from pathlib import Path
import json
import logging
import uuid

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "static" / "uploads"
MODEL_PATH = ROOT / "models" / "potato_cnn.keras"
LABELS_PATH = ROOT / "models" / "class_names.json"
MAX_BYTES = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
DISPLAY_NAMES = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
logging.basicConfig(level=logging.INFO)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
model = None
class_names = None


def load_model_once():
    """Load the disease model once and reuse it for requests."""
    global model, class_names
    if model is not None and class_names is not None:
        return True, None
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        return False, "The disease model is not ready. Prepare the dataset and run train.py first."
    try:
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        if labels != CLASS_NAMES:
            return False, "The saved model labels do not match the expected class order. Retrain the model."
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        if tuple(loaded_model.input_shape[1:]) != (128, 128, 3) or loaded_model.output_shape[-1] != 3:
            return False, "The saved model has an unexpected input or output shape. Retrain the model."
        model, class_names = loaded_model, labels
        return True, None
    except Exception:
        app.logger.exception("Could not load the disease model")
        return False, "The saved disease model could not be loaded. Check its files and retrain if needed."


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def home():
    ready, model_error = load_model_once()
    return render_template("index.html", model_ready=ready, model_error=model_error)


@app.errorhandler(413)
def too_large(_error):
    ready, model_error = load_model_once()
    return render_template(
        "index.html", error="That image is larger than the 8 MB upload limit.",
        model_ready=ready, model_error=model_error,
    ), 413


@app.post("/predict")
def predict():
    ready, model_error = load_model_once()
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return render_template("index.html", error="Choose an image file to continue.", model_ready=ready, model_error=model_error), 400
    if not allowed_file(uploaded.filename):
        return render_template("index.html", error="Use a PNG, JPG, or JPEG image.", model_ready=ready, model_error=model_error), 400
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded.stream)).convert("RGB")
        pixels = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)[None, ...]
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return render_template(
            "index.html", error="The uploaded file could not be read safely as an image.",
            model_ready=ready, model_error=model_error,
        ), 400

    if not ready:
        return render_template("index.html", error=model_error, model_ready=False, model_error=model_error), 503

    safe_name = f"{uuid.uuid4().hex}.jpg"
    saved_path = UPLOAD_DIR / safe_name
    preview = image.copy()
    preview.thumbnail((900, 900), Image.Resampling.LANCZOS)
    preview.save(saved_path, format="JPEG", quality=90)
    image_url = url_for("static", filename=f"uploads/{safe_name}")
    try:
        probabilities = model.predict(pixels, verbose=0)[0]
    except Exception:
        app.logger.exception("Disease prediction failed")
        saved_path.unlink(missing_ok=True)
        return render_template(
            "index.html", error="The image could not be analyzed. Please try another image.",
            model_ready=True, model_error=None,
        ), 500

    best_index = int(np.argmax(probabilities))
    ranked = sorted(zip(class_names, probabilities.tolist()), key=lambda item: item[1], reverse=True)
    result = {
        "label": DISPLAY_NAMES[class_names[best_index]],
        "score": float(probabilities[best_index]) * 100,
        "probabilities": [(DISPLAY_NAMES[name], float(score) * 100) for name, score in ranked],
        "image_url": image_url,
    }
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
