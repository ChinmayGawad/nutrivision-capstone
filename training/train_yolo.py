"""
Train YOLOv8 on the UECFOOD256 dataset (converted to YOLO format).

This script fine-tunes the YOLOv8 nano model on 256 food categories
from the UEC FOOD 256 dataset.

Usage:
    python training/train_yolo.py
"""

import os
from ultralytics import YOLO

# ── Configuration ──
DATA_YAML = os.path.join(os.path.dirname(__file__), "..", "datasets", "uecfood256_yolo", "data.yaml")
EPOCHS = 50          # Number of training epochs
IMG_SIZE = 640       # YOLOv8 default input size
BATCH_SIZE = 8       # Reduce if you get GPU OOM errors (try 4 or 2)
MODEL_NAME = "yolov8n.pt"  # Nano model (fastest, smallest)

def main():
    # Check if a previous checkpoint exists (for resuming after shutdown)
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "runs", "detect", "food_detector", "weights", "last.pt")
    # Also check numbered folders like food_detector2, food_detector3, etc.
    resume_path = None
    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "detect")
    if os.path.exists(runs_dir):
        for folder in sorted(os.listdir(runs_dir), reverse=True):
            if folder.startswith("food_detector"):
                candidate = os.path.join(runs_dir, folder, "weights", "last.pt")
                if os.path.exists(candidate):
                    resume_path = candidate
                    break

    print("=" * 50)
    print("YOLOv8 Food Detection Training")
    print("=" * 50)
    print(f"  Dataset: {os.path.abspath(DATA_YAML)}")
    print(f"  Epochs:  {EPOCHS}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")

    if resume_path:
        print(f"  RESUMING from: {resume_path}")
    else:
        print(f"  Base Model: {MODEL_NAME} (fresh start)")

    print("=" * 50)

    if resume_path:
        # Resume from the last checkpoint
        model = YOLO(resume_path)
        results = model.train(resume=True)
    else:
        # Fresh training from pre-trained YOLOv8 nano
        model = YOLO(MODEL_NAME)
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            name="food_detector",
            patience=10,         # Early stopping if no improvement for 10 epochs
            save=True,
            plots=True,
            verbose=True,
        )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("Best weights saved to: runs/detect/food_detector/weights/best.pt")
    print("=" * 50)

if __name__ == "__main__":
    main()
