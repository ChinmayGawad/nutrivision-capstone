import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os

print("Loading class names...")

# Load dataset info to get class names
_, ds_info = tfds.load('food101', with_info=True)
class_names = ds_info.features['label'].names
NUM_CLASSES = len(class_names)

print("Building model architecture...")

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

base_model.trainable = False  # MUST MATCH train.py exactly

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Explicitly build model
model.build((None, 224, 224, 3))

print("Loading trained weights...")

# Load weights
model.load_weights("food_model_weights.h5")

print("Model loaded successfully!")

# ----------------------------
# Image Path
# ----------------------------
# The test_images folder is located one directory up from the training folder
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "test_images", "pizza.jpg")

if not os.path.exists(IMAGE_PATH):
    print("Image not found. Check file path.")
    exit()

# ----------------------------
# Load and preprocess image
# ----------------------------
print("Processing image...")

img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((224, 224))

# EfficientNet expects inputs in the [0, 255] range
img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)

# ----------------------------
# Predict
# ----------------------------
print("Making prediction...")

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions[0])
confidence = float(np.max(predictions))

print("\n========== RESULT ==========")
print("Predicted Food :", class_names[predicted_class])
print("Confidence     :", round(confidence * 100, 2), "%")
print("============================")