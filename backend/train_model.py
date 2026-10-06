# train_model.py
import os, json, math, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# ------------ CONFIG ------------
DATA_DIR   = r"D:\coin1\images"   # ✅ Update to your actual dataset path
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 15                # train only top layers first
EPOCHS_FINETUNE = 15              # fine-tune deeper layers
SEED = 42
# --------------------------------

# Reproducibility
random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- Count images ----------
def count_per_class(root):
    counts = {}
    for name in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        folder = os.path.join(root, name)
        n = sum(1 for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
        counts[name] = n
    return counts

counts = count_per_class(DATA_DIR)
print("\n📸 Image counts per class:", counts)

# ---------- Dataset split ----------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",   # ✅ force 3 channels
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",   # ✅ same here
    shuffle=False
)


# Save label mapping
index_to_class = {i: name for i, name in enumerate(train_ds.class_names)}
with open("labels.json", "w") as f:
    json.dump(index_to_class, f, indent=2)
print("\n💾 Saved labels.json:", index_to_class)

# Prefetching for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ---------- Data augmentation ----------
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
], name="augmentation")

# ---------- Build model ----------
base = EfficientNetB0(include_top=False, weights="imagenet",
                      input_shape=(*IMG_SIZE, 3))
base.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = augment(inputs)
x = preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(index_to_class), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------- Callbacks ----------
ckpt = keras.callbacks.ModelCheckpoint(
    "best_model.keras", monitor="val_accuracy", mode="max",
    save_best_only=True, verbose=1
)
early = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", mode="max", patience=6,
    restore_best_weights=True, verbose=1
)
reduce = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)

# ---------- Class weights ----------
total = sum(counts.values())
num_classes = len(index_to_class)
class_weight = {}
for i, name in enumerate(index_to_class.values()):
    n = counts[name]
    class_weight[i] = total / (num_classes * max(1, n))
print("\n⚖️ Class weights:", class_weight)

# ---------- Train frozen backbone ----------
print("\n🚀 Training (frozen EfficientNet layers)...")
history1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weight,
    callbacks=[ckpt, early, reduce]
)

# ---------- Fine-tune deeper layers ----------
print("\n🎯 Fine-tuning deeper EfficientNet layers...")
base.trainable = True
for layer in base.layers[:-40]:  # keep first ~80% frozen
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # smaller LR for fine-tune
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight,
    callbacks=[ckpt, early, reduce]
)

# ---------- Save final models ----------
model.save("coin_model.keras")
print("\n✅ Training complete! Saved: best_model.keras, coin_model.keras, labels.json")
