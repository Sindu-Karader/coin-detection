from tensorflow import keras
import numpy as np

model = keras.models.load_model("coin_model.keras")
model.summary()

# Check a few random outputs
dummy = np.random.rand(1, 224, 224, 3).astype("float32")
out = model.predict(dummy)
print("\nOutput shape:", out.shape)
print("Probabilities:", out)
print("Sum of probs:", out.sum())
