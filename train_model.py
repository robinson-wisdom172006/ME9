"""
GET 324 Mini-Project (Group ME9)
Task: Binary image classification - Tomato Leaf Mold vs Tomato Septoria Leaf Spot
Approach: Transfer learning using MobileNetV2 (pre-trained on ImageNet)

HOW TO GET THE DATASET
-----------------------
1. Go to Kaggle and search "PlantVillage Dataset" (e.g. emmarex/plantdisease).
   Direct link: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Download and unzip it.
3. Inside, find these two folders (names may vary slightly by version):
   - Tomato___Leaf_Mold
   - Tomato___Septoria_leaf_spot
4. Copy ONLY those two folders into a new folder structure like this,
   in the same directory as this script:

   dataset/
       Tomato_Leaf_Mold/        <- put all Leaf Mold images here
       Tomato_Septoria_Leaf_Spot/  <- put all Septoria images here

5. Run this script:  python train_model.py

The script will:
- Split the images into train/validation sets automatically (80/20)
- Fine-tune a MobileNetV2 model on the two classes
- Save the trained model as "tomato_model.h5" (used by app.py)
- Save training curves as "training_history.png" for your report
"""

import os
import matplotlib
matplotlib.use("Agg")  # avoids display backend issues (useful on Kali/headless setups)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# CONFIG
# ----------------------------
DATASET_DIR = "/home/mechamodella/Documents/DataSet"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_OUTPUT = "tomato_model.h5"
CLASS_NAMES_FILE = "class_names.txt"

# ----------------------------
# 1. DATA LOADING + AUGMENTATION
# ----------------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
)

# Save class name mapping (index -> label) so app.py knows what "0" and "1" mean
class_indices = train_generator.class_indices  # e.g. {'Tomato_Leaf_Mold': 0, 'Tomato_Septoria_Leaf_Spot': 1}
index_to_class = {v: k for k, v in class_indices.items()}
with open(CLASS_NAMES_FILE, "w") as f:
    for i in range(len(index_to_class)):
        f.write(index_to_class[i] + "\n")

print("Class mapping:", class_indices)

# ----------------------------
# 2. BUILD MODEL (Transfer Learning)
# ----------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze base for initial training

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),  # binary classification
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ----------------------------
# 3. TRAIN (frozen base)
# ----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop],
)

# ----------------------------
# 4. FINE-TUNE (unfreeze top layers of base model)
# ----------------------------
base_model.trainable = True
# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop],
)

# ----------------------------
# 5. EVALUATE + SAVE
# ----------------------------
loss, acc = model.evaluate(val_generator)
print(f"Final validation accuracy: {acc*100:.2f}%")
print(f"Final validation loss: {loss:.4f}")

model.save(MODEL_OUTPUT)
print(f"Model saved to {MODEL_OUTPUT}")

# ----------------------------
# 6. PLOT TRAINING CURVES (for report/screenshots)
# ----------------------------
acc_all = history.history["accuracy"] + fine_tune_history.history["accuracy"]
val_acc_all = history.history["val_accuracy"] + fine_tune_history.history["val_accuracy"]
loss_all = history.history["loss"] + fine_tune_history.history["loss"]
val_loss_all = history.history["val_loss"] + fine_tune_history.history["val_loss"]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc_all, label="Train Accuracy")
plt.plot(val_acc_all, label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_all, label="Train Loss")
plt.plot(val_loss_all, label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
print("Training curves saved to training_history.png")
