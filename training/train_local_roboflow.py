import os
import sys
import torch
from ultralytics import YOLO

# ─── Training Settings ───
EPOCHS = 150           # Number of epochs to train
BATCH_SIZE = 32        # Lower this to 4 or 2 if your PC gets an "Out of Memory" error
IMG_SIZE = 640        # Standard image size
# The model currently working on your project:
MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "detect", "food_detector5", "weights", "best.pt")
 
def main():
    print("=" * 60)
    print("🍽️ NutriVision AI - Local Indian Food Model Training")
    print("=" * 60)

    print("\n[1/3] Downloading Dataset from Roboflow...")
    try:
        from roboflow import Roboflow
        # Using the NEW dataset snippet you provided
        rf = Roboflow(api_key="xIwAT1xq8QtTdCCT1PVd")
        project = rf.workspace("microplastics-vypxl").project("indian-food-txofy")
        version = project.version(6)
        dataset = version.download("yolov8")
        
        dataset_path = dataset.location
        print(f"✅ Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        sys.exit(1)

    # 2. Verify data.yaml exists
    print("\n[2/3] Preparing Training...")
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: data.yaml not found at {data_yaml_path}")
        sys.exit(1)
        
    print(f"  📂 Dataset Data: {data_yaml_path}")
    print(f"  🧠 Model:        {MODEL_NAME}")
    
    # 3. Detect GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  🚀 GPU:          {gpu_name} ({gpu_vram:.1f} GB VRAM)")
    else:
        print("  ⚠️ No CUDA GPU detected — training will use CPU (very slow).")

    model_run_name = "indian_food_v2"
    last_checkpoint = os.path.join(".", "runs", "detect", model_run_name, "weights", "last.pt")
    
    if os.path.exists(last_checkpoint):
        print("\n[3/3] 🔄 Found previous training checkpoint!")
        print(f"Resuming training from: {last_checkpoint}...")
        model = YOLO(last_checkpoint)
        results = model.train(resume=True)
    else:
        print("\n[3/3] Starting Fresh Training on your PC...")
        print("You can monitor GPU usage in Task Manager during this step.\n")
        
        model = YOLO(MODEL_NAME)
        
        results = model.train(
            data=data_yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            name=model_run_name,
            patience=15,
            save=True,
            device=device,
            save_period=1, # Explicitly save weights every single epoch
        )
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("Your new model weights have been saved.")
    print("To use the new model in your backend, copy the 'best.pt' file from:")
    print(f"  runs/detect/{model_run_name}/weights/best.pt")
    print("into the main folder of your backend.")
    print("=" * 60)

if __name__ == "__main__":
    main()
