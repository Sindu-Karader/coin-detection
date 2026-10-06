import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("coin_model.h5")



# Print model summary to confirm it loaded
model.summary()
