import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Path to dataset
data_dir = r"D:\coin\images"

# Load dataset with automatic labeling
img_height, img_width = 128, 128
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="categorical",  # one-hot encoding
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Show class names
print("Classes found:", train_ds.class_names)

# Show few sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i].numpy().argmax()])
        plt.axis("off")
plt.show()
