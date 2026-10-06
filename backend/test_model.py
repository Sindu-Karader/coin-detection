import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
from coin_info import coin_info
import pytesseract

# Point to installed exe (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



MODEL = "best_model.keras" if os.path.exists("best_model.keras") else "coin_model.keras"
IMG_SIZE = (224, 224)

# Load model + labels
model = keras.models.load_model(MODEL)
with open("labels.json", "r") as f:
    index_to_class = json.load(f)
class_labels = [index_to_class[str(i)] for i in range(len(index_to_class))]
print("Loaded model:", MODEL)
print("Classes:", class_labels)

def load_for_model(path):
    img = keras.utils.load_img(path)                 # original size
    x = keras.utils.img_to_array(img)
    x = tf.image.resize(x, IMG_SIZE)                 # same resize as training
    x = preprocess_input(x)
    return tf.expand_dims(x, 0)

# Simple test-time augmentation: flip + small rotate (averaged)
def predict_tta(x):
    variants = [x]
    variants.append(tf.image.flip_left_right(x))
    # small rotation via affine grid (approx using keras preprocessing layer)
    rot_layer = keras.layers.RandomRotation(0.05)
    variants.append(rot_layer(x, training=True))
    preds = [model.predict(v, verbose=0)[0] for v in variants]
    return np.mean(preds, axis=0)

img_path = input("Enter image path: ").strip().replace("\\", "/")
if not os.path.exists(img_path):
    print("❌ File not found:", img_path)
else:
    # Show the exact image you provided
    plt.imshow(keras.utils.load_img(img_path))
    plt.title("Input")
    plt.axis("off")
    plt.show(block=False); plt.pause(1.5); plt.close()

    x = load_for_model(img_path)
    probs = predict_tta(x)
    top3 = np.argsort(probs)[-3:][::-1]

    print("\n🔎 Top predictions:")
    for i in top3:
        print(f"  {class_labels[i]}: {probs[i]*100:.2f}%")
    print(f"\n✅ Final: {class_labels[top3[0]]} ({probs[top3[0]]*100:.2f}% confidence)")


    final_label = class_labels[top3[0]]

# Show extra coin details if available
if final_label in coin_info:
    details = coin_info[final_label]
    print("\n📌 Extra Info:")
    print(f"  Country    : {details['country']}")
    print(f"  Year       : {details['year']}")
    print(f"  Material   : {details['material']}")
    print(f"  Description: {details['description']}")

