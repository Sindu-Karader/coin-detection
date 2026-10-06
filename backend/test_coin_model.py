import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("final_coin_model.h5")

# Load image
img = cv2.imread("test_coin.jpg")
img_resized = cv2.resize(img, (128, 128))
img_array = np.expand_dims(img_resized/255.0, axis=0)

# Predict
pred = model.predict(img_array)
class_index = np.argmax(pred)
class_labels = list(model.class_names) if hasattr(model, "class_names") else None

print("Prediction:", class_labels[class_index] if class_labels else class_index)
