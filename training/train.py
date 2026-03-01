import os
import sys

# --- GPU DLL Fix for Windows + TF 2.10 ---
# Tell the Windows DLL loader exactly where CUDA/cuDNN libraries are (pip-installed via nvidia packages)
_venv_base = os.path.dirname(os.path.dirname(sys.executable))  # venv root
_nvidia_base = os.path.join(_venv_base, "Lib", "site-packages", "nvidia")
for _pkg in ["cuda_runtime", "cudnn", "cublas", "cuda_nvrtc", "cufft", "curand", "cusolver", "cusparse"]:
    _bin = os.path.join(_nvidia_base, _pkg, "bin")
    if os.path.isdir(_bin):
        os.add_dll_directory(_bin)

import pandas as pd
import numpy as np
import tensorflow as tf

print("Preparing to load Nutrition5k dataset...")

# Paths
DATASET_DIR = r"d:\Cap Stone\food-recognition-project\Dataset"
CSV_PATH = os.path.join(DATASET_DIR, "dish_ingredients.csv")
IMAGES_DIR = os.path.join(DATASET_DIR, "imagery")

# 1. Load and aggregate CSV
print("Aggregating dish macros from CSV...")
df = pd.read_csv(CSV_PATH)

# Group by dish_id to get total macros per dish
df_dish = df.groupby('dish_id')[['grams', 'calories', 'fat', 'carb', 'protein']].sum().reset_index()

# 2. Match images with labels
print("Matching images with labels...")
image_paths = []
labels = []

# We only include dishes where the image actually exists on disk
valid_count = 0
for _, row in df_dish.iterrows():
    dish_id = row['dish_id']
    # The Nutrition5k dataset images are stored as imagery/realsense_overhead/dish_xxx/rgb.png
    img_path = os.path.join(IMAGES_DIR, "realsense_overhead", dish_id, "rgb.png")
    
    # We delay the os.path.exists check until the download is actually done, 
    # but we can filter the list in python to be safe
    if os.path.exists(img_path):
        image_paths.append(img_path)
        labels.append([row['calories'], row['grams'], row['fat'], row['carb'], row['protein']])
        valid_count += 1
    else:
        # Fallback to side angles if overhead is missing
        backup_path = os.path.join(IMAGES_DIR, "side_angles", dish_id, "rgb.png")
        if os.path.exists(backup_path):
            image_paths.append(backup_path)
            labels.append([row['calories'], row['grams'], row['fat'], row['carb'], row['protein']])
            valid_count += 1

image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.float32)

print(f"Total dishes mapped from CSV: {len(labels)}")

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# 3. Setup tf.data.Dataset
def load_and_preprocess_image(path, label):
    # Read image
    image = tf.io.read_file(path)
    # Decode and ensure 3 channels
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # EfficientNet preprocessing
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

# Create pipeline
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.shuffle(buffer_size=1000, seed=42)
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

# Splitting 80/20
DATASET_SIZE = len(labels)
train_size = int(0.8 * DATASET_SIZE)

ds_train = dataset.take(train_size).batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_test = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Dataset pipeline ready.")

# 4. Build Model (always build architecture; load weights if checkpoint exists)
print("Loading EfficientNetB0...")
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Fine-tune the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Build Model for REGRESSION (predicting macros)
# Output is 5 continuous values: Calories, Mass (grams), Fat, Carb, Protein
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    # Use ReLU activation because macros/calories can never be negative
    tf.keras.layers.Dense(5, activation='relu') 
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# Load saved weights if checkpoint exists (resume from where we left off)
checkpoint_path = r"d:\Cap Stone\food-recognition-project\best_macro_model.weights.h5"
if os.path.exists(checkpoint_path):
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    model.load_weights(checkpoint_path)
else:
    print("No checkpoint found. Starting training from scratch.")

print("Starting deep regression training...")

# --- [Advanced Capstone Deep Learning Callbacks] ---
# 1. Early Stopping: Stop training if the AI stops improving (prevents wasting time)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

# 2. Reduce LR on Plateau: If the AI gets stuck, slow down the learning rate to make finer adjustments
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=5, 
    min_lr=0.00001,
    verbose=1
)

# 3. Model Checkpoint: Custom callback to avoid TF 2.10 EagerTensor JSON serialization bug
class BestModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = float(logs.get('val_loss', float('inf')))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save_weights(self.filepath)
            print(f"\nEpoch {epoch+1}: val_loss improved to {val_loss:.5f}, saving weights to {self.filepath}")

checkpoint = BestModelCheckpoint(r"d:\Cap Stone\food-recognition-project\best_macro_model.weights.h5")

# Uncomment below to actually train the model. 
# We raised the epochs to 100 to allow the AI to actually learn. The EarlyStopping will stop it if it finishes early.
model.fit(
    ds_train, 
    epochs=100, 
    validation_data=ds_test,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

print("Training completed. The best model was saved as best_macro_model.weights.h5!")
model.save_weights(r"d:\Cap Stone\food-recognition-project\macro_model.weights.h5")
print("Final backup model weights saved successfully as macro_model.weights.h5!")
