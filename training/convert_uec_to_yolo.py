"""
Convert UECFOOD256 dataset to YOLOv8 format.

UECFOOD256 format:
  - 256 folders (1-256), each with .jpg images and bb_info.txt
  - bb_info.txt: "img x1 y1 x2 y2"  (pixel coordinates, absolute)

YOLOv8 format:
  - images/train/, images/val/   (image files)
  - labels/train/, labels/val/   (one .txt per image: "class x_center y_center width height" normalized 0-1)
  - data.yaml
"""

import os, shutil, random
from PIL import Image

# ── Configuration ──
SRC_DIR   = os.path.join(os.path.dirname(__file__), "..", "Dataset", "UECFOOD256")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "datasets", "uecfood256_yolo")
VAL_SPLIT = 0.15   # 15% of images go to validation

random.seed(42)

# ── 1. Parse category.txt to get class names ──
cat_file = os.path.join(SRC_DIR, "category.txt")
class_names = {}
with open(cat_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("id"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            class_names[int(parts[0])] = parts[1]

num_classes = len(class_names)
print(f"Found {num_classes} food classes.")

# ── 2. Create output directories ──
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

# ── 3. Process each category folder ──
total_images = 0
total_labels = 0

for class_id in sorted(class_names.keys()):
    class_dir = os.path.join(SRC_DIR, str(class_id))
    bb_file = os.path.join(class_dir, "bb_info.txt")

    if not os.path.exists(bb_file):
        print(f"  [SKIP] No bb_info.txt in folder {class_id}")
        continue

    # Parse bounding boxes
    bboxes = {}  # img_name -> list of (x1, y1, x2, y2)
    with open(bb_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("img"):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            img_id, x1, y1, x2, y2 = parts
            img_name = f"{img_id}.jpg"
            if img_name not in bboxes:
                bboxes[img_name] = []
            bboxes[img_name].append((int(x1), int(y1), int(x2), int(y2)))

    # Process images that have bounding boxes
    img_names = list(bboxes.keys())
    random.shuffle(img_names)
    val_count = max(1, int(len(img_names) * VAL_SPLIT))
    val_imgs = set(img_names[:val_count])

    for img_name in img_names:
        img_path = os.path.join(class_dir, img_name)
        if not os.path.exists(img_path):
            continue

        # Get image dimensions
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception:
            continue

        # Decide split
        split = "val" if img_name in val_imgs else "train"

        # Copy image with unique name: classid_imagename
        unique_name = f"{class_id}_{img_name}"
        dst_img = os.path.join(OUT_DIR, "images", split, unique_name)
        shutil.copy2(img_path, dst_img)
        total_images += 1

        # Write YOLO label file
        label_name = unique_name.replace(".jpg", ".txt")
        dst_label = os.path.join(OUT_DIR, "labels", split, label_name)
        with open(dst_label, "w") as lf:
            for (x1, y1, x2, y2) in bboxes[img_name]:
                # Convert to YOLO normalized format
                # class_index is 0-based (class_id is 1-based)
                cls_idx = class_id - 1
                x_center = ((x1 + x2) / 2.0) / img_w
                y_center = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                # Clamp values to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                lf.write(f"{cls_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                total_labels += 1

    if class_id % 50 == 0:
        print(f"  Processed {class_id}/{num_classes} classes...")

# ── 4. Generate data.yaml ──
yaml_path = os.path.join(OUT_DIR, "data.yaml")
with open(yaml_path, "w", encoding="utf-8") as yf:
    yf.write(f"path: {os.path.abspath(OUT_DIR)}\n")
    yf.write("train: images/train\n")
    yf.write("val: images/val\n\n")
    yf.write(f"nc: {num_classes}\n")
    yf.write("names: [")
    name_list = []
    for i in sorted(class_names.keys()):
        # Sanitize: remove apostrophes and ampersands that break YAML
        safe_name = class_names[i].replace("'", "").replace("&", "and")
        name_list.append(f'"{safe_name}"')
    yf.write(", ".join(name_list))
    yf.write("]\n")

print(f"\n{'='*50}")
print(f"CONVERSION COMPLETE!")
print(f"  Total images copied: {total_images}")
print(f"  Total bounding boxes: {total_labels}")
print(f"  Output directory: {os.path.abspath(OUT_DIR)}")
print(f"  data.yaml: {os.path.abspath(yaml_path)}")
print(f"{'='*50}")
