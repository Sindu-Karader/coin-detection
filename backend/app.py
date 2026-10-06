from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os, numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ allow all origins

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- dummy model logic for testing ---
class_labels = ["1_rupee", "2_rupee", "5_rupee", "10_rupee"]

def load_for_model(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def predict_tta(x):
    p = np.random.rand(len(class_labels))
    return p / p.sum()

@app.route("/detect_coin", methods=["POST"])
def detect_coin():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        Image.open(filepath).verify()
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    x = load_for_model(filepath)
    probs = predict_tta(x)
    idx = int(np.argmax(probs))
    result = {"coin": class_labels[idx], "confidence": float(probs[idx])}
    return jsonify(result)


if __name__ == "__main__":
    # ✅ Make Flask reachable from your frontend (localhost or 127)
    app.run(host="0.0.0.0", port=5000, debug=True)
