import os
import csv as _csv
import time
import jwt
import requests
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Depends, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import numpy as np
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

# In a real environment, load this safely from an environment variable!
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "my_super_secret_jwt_key")
ALGORITHM = "HS256"

# Edamam Nutrition API Keys (Demo/Free Tier)
EDAMAM_APP_ID = "cc3fe44a" # Note for student: You can replace these with your own keys from Edamam
EDAMAM_APP_KEY = "6be30bfa2e95a94ee8997ef0d195a6ec"

# 1. Initialize FastAPI & Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Food Nutrition Estimator API")
app.state.limiter = limiter
app.add_exception_handler(429, lambda request, exc: JSONResponse(
    status_code=429, content={"detail": "Too many requests. Please slow down."}
))

# 2. Add Security Middleware
app.add_middleware(SlowAPIMiddleware) # CyberSecurity: Prevents DDoS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://YOUR-FRONTEND-APP-NAME.netlify.app", "*"], # Added * temporarily for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- [YOLOv8 Custom Food Detection Model (256 Classes)] ---
# Trained on UECFOOD256 dataset: rice, pizza, sushi, ramen, steak, curry, etc.
# Falls back to generic yolov8n.pt if custom weights are not found.

CUSTOM_YOLO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "detect", "food_detector5", "weights", "best.pt")
FALLBACK_YOLO = "yolov8n.pt"

try:
    from ultralytics import YOLO
    if os.path.exists(CUSTOM_YOLO_PATH):
        print(f"[YOLO] Loading CUSTOM trained food model (256 classes)...")
        yolo_model = YOLO(CUSTOM_YOLO_PATH)
        print("[YOLO] Custom food model loaded successfully!")
    else:
        print(f"[YOLO] Custom model not found. Loading generic {FALLBACK_YOLO}...")
        yolo_model = YOLO(FALLBACK_YOLO)
        print("[YOLO] Generic model loaded.")
    USE_YOLO = True
except ImportError:
    print("[YOLO] ultralytics not installed. Multi-food detection disabled.")
    yolo_model = None
    USE_YOLO = False
except Exception as e:
    print(f"[YOLO] Error loading YOLO model: {e}")
    yolo_model = None
    USE_YOLO = False


# --- [Built-in Nutrition Database — Nutrition5k Dataset + Manual Overrides] ---
# Dynamically loaded from ingredients_metadata.csv (555 real foods from our dataset!)
# Values in the CSV are per gram; we convert to per 100g for consistency.
def _load_nutrition_db():
    db = {}
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Dataset", "ingredients_metadata.csv")
    )
    if os.path.exists(csv_path):
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    name = row.get("ingr_name", "").strip().lower()
                    try:
                        # CSV columns: cal/g, fat(g), carb(g), protein(g)  — per gram
                        cal   = float(row.get("cal/g", 0) or 0) * 100
                        fat   = float(row.get("fat(g)", 0) or 0) * 100
                        carbs = float(row.get("carb(g)", 0) or 0) * 100
                        prot  = float(row.get("protein(g)", 0) or 0) * 100
                        if name and cal > 0:
                            db[name] = {"calories": round(cal, 1), "fat": round(fat, 1),
                                        "carbs": round(carbs, 1), "protein": round(prot, 1)}
                    except (ValueError, TypeError):
                        continue
            print(f"[NutritionDB] Loaded {len(db)} foods from Nutrition5k CSV")
        except Exception as e:
            print(f"[NutritionDB] Could not read CSV: {e}")
    else:
        print(f"[NutritionDB] ingredients_metadata.csv not found — using manual DB only")

    # Manual overrides / items not in Nutrition5k (egg varieties, Indian foods, etc.)
    manual = {
        "egg":            {"calories": 155, "fat": 11.0, "carbs": 1.1,  "protein": 13.0},
        "boiled egg":     {"calories": 155, "fat": 11.0, "carbs": 1.1,  "protein": 13.0},
        "fried egg":      {"calories": 196, "fat": 15.0, "carbs": 0.8,  "protein": 14.0},
        "omelette":       {"calories": 154, "fat": 11.0, "carbs": 0.9,  "protein": 11.0},
        "omelet":         {"calories": 154, "fat": 11.0, "carbs": 0.9,  "protein": 11.0},
        "egg omelette":   {"calories": 154, "fat": 11.0, "carbs": 0.9,  "protein": 11.0},
        "egg omelet":     {"calories": 154, "fat": 11.0, "carbs": 0.9,  "protein": 11.0},
        "poached egg":    {"calories": 143, "fat": 9.5,  "carbs": 0.7,  "protein": 13.0},
        "pizza":          {"calories": 266, "fat": 10.0, "carbs": 33.0, "protein": 11.0},
        "burger":         {"calories": 295, "fat": 14.0, "carbs": 24.0, "protein": 17.0},
        "roti":           {"calories": 297, "fat": 3.7,  "carbs": 60.0, "protein": 9.0},
        "chapati":        {"calories": 297, "fat": 3.7,  "carbs": 60.0, "protein": 9.0},
        "dal":            {"calories": 116, "fat": 0.4,  "carbs": 20.0, "protein": 8.0},
        "idli":           {"calories": 58,  "fat": 0.4,  "carbs": 11.0, "protein": 2.0},
        "dosa":           {"calories": 133, "fat": 2.7,  "carbs": 22.0, "protein": 4.0},
        "biryani":        {"calories": 163, "fat": 5.0,  "carbs": 22.0, "protein": 8.0},
        "paneer":         {"calories": 265, "fat": 20.0, "carbs": 3.6,  "protein": 18.0},
        "french fries":   {"calories": 312, "fat": 15.0, "carbs": 41.0, "protein": 3.4},
        "donut":          {"calories": 452, "fat": 25.0, "carbs": 51.0, "protein": 5.0},
        "pancake":        {"calories": 227, "fat": 8.0,  "carbs": 36.0, "protein": 6.0},
        "hot dog":        {"calories": 290, "fat": 17.0, "carbs": 22.0, "protein": 11.0},
        # Japanese Foods from UECFOOD256 for Demo
        "okinawa soba":   {"calories": 420, "fat": 12.0, "carbs": 60.0, "protein": 18.0},
        "goya chanpuru":  {"calories": 240, "fat": 15.0, "carbs": 12.0, "protein": 14.0},
        "sushi":          {"calories": 350, "fat": 2.0,  "carbs": 75.0, "protein": 11.0},
        "ramen":          {"calories": 436, "fat": 16.0, "carbs": 55.0, "protein": 15.0},
        "takoyaki":       {"calories": 310, "fat": 14.0, "carbs": 38.0, "protein": 9.0},
        "okonomiyaki":    {"calories": 520, "fat": 22.0, "carbs": 65.0, "protein": 17.0},
        "udon":           {"calories": 280, "fat": 2.0,  "carbs": 55.0, "protein": 10.0},
        "curry":          {"calories": 480, "fat": 20.0, "carbs": 60.0, "protein": 12.0},
        "tempura":        {"calories": 320, "fat": 20.0, "carbs": 30.0, "protein": 6.0},
        "yakitori":       {"calories": 150, "fat": 6.0,  "carbs": 2.0,  "protein": 22.0},
        "tonkatsu":       {"calories": 550, "fat": 35.0, "carbs": 30.0, "protein": 25.0},
        "miso soup":      {"calories": 40,  "fat": 1.5,  "carbs": 4.0,  "protein": 3.0},
        "gyudon":         {"calories": 650, "fat": 25.0, "carbs": 85.0, "protein": 22.0},
        "karaage":        {"calories": 290, "fat": 18.0, "carbs": 12.0, "protein": 16.0},
    }
    db.update(manual)   # manual entries win on overlap
    return db

NUTRITION_DB = _load_nutrition_db()
print(f"[NutritionDB] Total entries available: {len(NUTRITION_DB)}")

def lookup_local_nutrition(food_query: str, serving_size_g: float):
    """
    Fast built-in lookup — no API key needed.
    Matches food_query against our local USDA-based table, scales to serving size.
    """
    query_lower = food_query.strip().lower()
    matched = None

    # Exact match first
    if query_lower in NUTRITION_DB:
        matched = NUTRITION_DB[query_lower]
    else:
        # Partial match: only use if the query is a single word or if it's a very close match
        # This prevents "chicken rice" from matching raw "chicken" (which has 0 carbs)
        for key, values in NUTRITION_DB.items():
            if key == query_lower or (len(query_lower.split()) == 1 and (key in query_lower or query_lower in key)):
                matched = values
                print(f"[LocalDB] Partial match: '{query_lower}' -> '{key}'")
                break
        
        if not matched:
            print(f"[LocalDB] No safe local match found for '{query_lower}'. Will try API.")

    if matched:
        scale = serving_size_g / 100.0
        return {
            "calories": round(matched["calories"] * scale, 1),
            "mass_grams": serving_size_g,
            "fat_grams": round(matched["fat"] * scale, 1),
            "carbs_grams": round(matched["carbs"] * scale, 1),
            "protein_grams": round(matched["protein"] * scale, 1),
        }
    return None


# --- [NLP Extra Information API — Edamam fallback] ---
def fetch_edamam_nutrition(food_query: str, serving_size_g: float = None):
    """
    Tier 1: Built-in local nutrition DB (always works, no API key).
    Tier 2: Edamam API (requires valid credentials).
    """
    mass = serving_size_g or 100.0

    # --- Tier 1: Local DB ---
    local = lookup_local_nutrition(food_query, mass)
    if local:
        print(f"[LocalDB] Hit: '{food_query}' at {mass}g -> {local}")
        return local

    # --- Tier 2: Edamam (fallback for unlisted foods) ---
    url = (
        f"https://api.edamam.com/api/nutrition-details"
        f"?app_id={EDAMAM_APP_ID}&app_key={EDAMAM_APP_KEY}"
    )
    query_str = f"{int(mass)}g {food_query}"
    payload = {"title": food_query, "ingr": [query_str]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=8)
        print(f"[Edamam] Status: {response.status_code} | Query: {query_str}")

        if response.status_code == 200:
            data = response.json()
            calories = data.get("calories", 0)
            if calories > 0:
                nutrients = data.get("totalNutrients", {})
                return {
                    "calories": round(calories, 1),
                    "mass_grams": round(data.get("totalWeight", mass), 1),
                    "fat_grams": round(nutrients.get("FAT", {}).get("quantity", 0), 1),
                    "carbs_grams": round(nutrients.get("CHOCDF", {}).get("quantity", 0), 1),
                    "protein_grams": round(nutrients.get("PROCNT", {}).get("quantity", 0), 1)
                }
            else:
                print(f"[Edamam] No calorie data returned.")
        else:
            print(f"[Edamam] Error {response.status_code}: {response.text[:200]}")
    except requests.exceptions.Timeout:
        print("[Edamam] Timed out.")
    except Exception as e:
        print(f"[Edamam] Exception: {e}")

    return None


# ----------------------------------------
# 3. Security: JWT Authentication Helpers
# ----------------------------------------
def verify_token(req: Request):
    """
    CyberSecurity Requirement:
    Ensure the user is authorized before they can upload images.
    """
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ----------------------------------------
# 4. API Endpoints
# ----------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Food Nutrition API is running"}

@app.post("/login")
@limiter.limit("10/minute") # Strict limit on login endpoints
def login(request: Request):
    """Mock Login to generate a JWT for testing"""
    # Expiration time: 1 hour
    expiration = time.time() + 3600
    token = jwt.encode({"user_id": "demo_user", "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/predict")
@limiter.limit("5/minute") # Limit image uploads per IP to prevent spam
async def predict_nutrition(
    request: Request,
    file: UploadFile = File(...),
    food_name: Optional[str] = Form(None),
    serving_size: Optional[float] = Form(None),
    serving_unit: Optional[str] = Form(None)
    # Uncomment to enforce JWT token requirement:
    # user = Depends(verify_token) 
):
    """
    Core AI Endpoint + Multimodal NLP
    Upload food image (+ optional text) -> Detect macros -> Show Calories
    """
    # 1. CyberSecurity Check: Validate File Type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")

    # 2. Read Image Data
    image_bytes = await file.read()
    
    # CyberSecurity Check: Validate File Size (Max 5MB)
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 5MB allowed.")

    # 3. Object Detection (YOLOv8)
    try:
        detected_foods = []
        
        # Check if the user bypassed AI via Manual Text Input
        if food_name and food_name.strip() != "":
            detected_foods.append(food_name.strip())
            print(f"[YOLO] User override: {food_name}")
        elif USE_YOLO and yolo_model is not None:
            try:
                print("[YOLO] Scanning for multiple foods...")
                # We must save the bytes to a temp file for Ultralytics 
                # (YOLO prefers file paths or PIL Images)
                img_clf = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Run YOLO prediction with higher confidence threshold and NMS (Non-Maximum Suppression)
                # conf=0.50 means the AI must be 50% sure it's correct (reduces false positives)
                # iou=0.45 prevents overlapping bounding boxes for the same food item
                results = yolo_model(img_clf, conf=0.50, iou=0.45, verbose=False)
                
                # Loop through all bounding boxes found
                for box in results[0].boxes:
                    cls_name = yolo_model.names[int(box.cls)].replace("_", " ")
                    detected_foods.append(cls_name)
                
                # Remove duplicate detections (same food detected multiple times)
                detected_foods = list(set(detected_foods))
                
                if detected_foods:
                    print(f"[YOLO] Found foods: {detected_foods}")
                else:
                    print("[YOLO] No confident food objects detected.")
            except Exception as e:
                print(f"[YOLO] Error running object detection: {e}")

        # 4. Nutrition Calculation Loop
        # We will sum the macros for ALL detected foods!
        total_macros = {
            "calories": 0.0,
            "mass_grams": 0.0, 
            "fat_grams": 0.0,
            "carbs_grams": 0.0,
            "protein_grams": 0.0
        }
        
        # Determine the serving size per item
        # If the user provided a total manual weight, distribute it evenly.
        # Otherwise, default to 150g per detected item.
        items_count = len(detected_foods) if len(detected_foods) > 0 else 1
        per_item_mass_g = 150.0 # Default fallback
        
        if serving_size is not None:
            if serving_unit == "oz":
                per_item_mass_g = (serving_size * 28.3495) / items_count
            elif serving_unit == "lbs":
                per_item_mass_g = (serving_size * 453.592) / items_count
            else: # Defaults to grams
                per_item_mass_g = serving_size / items_count

        data_source_used = "Unknown"

        # 5. Look up each detected food and sum the macros
        if not detected_foods:
            # If no objects detected, provide safe default
            detected_foods = ["Unknown (AI Vision Failed)"]
            data_source_used = "Generic Fallback"
            total_macros = {
                "calories": 250.0,
                "mass_grams": per_item_mass_g,
                "fat_grams": 10.0,
                "carbs_grams": 25.0,
                "protein_grams": 15.0
            }
        else:
            data_source_used = "YOLOv8 + Nutrition DB"
            total_mass_g = 0.0
            
            for food in detected_foods:
                # 1. Try local DB first. 2. Fall back to Edamam API if not found.
                item_macros = fetch_edamam_nutrition(food_query=food, serving_size_g=per_item_mass_g)
                
                if item_macros:
                    total_macros["calories"] += item_macros["calories"]
                    total_macros["fat_grams"] += item_macros["fat_grams"]
                    total_macros["carbs_grams"] += item_macros["carbs_grams"]
                    total_macros["protein_grams"] += item_macros["protein_grams"]
                    total_mass_g += item_macros.get("mass_grams", per_item_mass_g)
                else:
                    print(f"[YOLO] Warning: {food} detected but not found in DB.")
            
            total_macros["mass_grams"] = total_mass_g
            
            # Safety check: if somehow it detected foods but ALL lookups failed, use fallback
            if total_macros["calories"] < 1:
                 print("[YOLO] AI prediction was 0cals. Using generic placeholder.")
                 total_macros = {
                    "calories": 250.0,
                    "mass_grams": 150.0,
                    "fat_grams": 10.0,
                    "carbs_grams": 25.0,
                    "protein_grams": 15.0
                }
                 data_source_used = "Generic Fallback (Lookups Failed)"

        # Join the list of detected foods into a nice string for the UI banner
        final_food_name = ", ".join([f.capitalize() for f in detected_foods])
        
        return {
            "status": "success",
            "filename": file.filename,
            "detected_food_name": final_food_name,
            "data_source_used": data_source_used,
            "nutrition_estimates": {
                "calories": round(total_macros["calories"], 1),
                "mass_grams": round(total_macros["mass_grams"], 1),
                "fat_grams": round(total_macros["fat_grams"], 1),
                "carbs_grams": round(total_macros["carbs_grams"], 1),
                "protein_grams": round(total_macros["protein_grams"], 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run the Cloud server
    uvicorn.run(app, host="0.0.0.0", port=8000)