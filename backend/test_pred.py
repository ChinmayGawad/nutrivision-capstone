import os
import tensorflow as tf
import numpy as np

IMG_SIZE = 224
WEIGHTS_PATH = r"d:\Cap Stone\food-recognition-project\best_macro_model.weights.h5"

print("Building model architecture...")
_base = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights=None # we don't need to download imagenet just to check weights
)

for layer in _base.layers[:-30]:
    layer.trainable = False
for layer in _base.layers[-30:]:
    layer.trainable = True

macro_model = tf.keras.Sequential([
    _base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation='relu')
])

print("Dummy pass...")
macro_model(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]))

print("Loading weights...")
try:
    macro_model.load_weights(WEIGHTS_PATH, by_name=True)
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")

print("Testing 5 random images...")
for i in range(5):
    random_img = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3) * 255
    random_img = tf.keras.applications.efficientnet.preprocess_input(random_img)
    pred = macro_model.predict(random_img, verbose=0)
    print(f"Prediction {i+1}: {pred}")

# Testing real output of final dense layer weights
dense_layer = macro_model.layers[-1]
weights, biases = dense_layer.get_weights()
print(f"Dense layer bias: {biases}")
