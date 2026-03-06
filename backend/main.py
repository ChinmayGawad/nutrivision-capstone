import os
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
import tensorflow as tf
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
    allow_origins=["*"], # Allow UI frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- [Image Classification Model Loading] ---
print("Loading MobileNetV2 Classifier for Food Recognition...")
try:
    classifier_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    print("MobileNetV2 Classifier loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load MobileNetV2: {e}")
    classifier_model = None

# --- [Deep Learning Regression Model Loading] ---
print("Loading Trained Deep Learning Model...")
# Path is relative to this script — works on Windows locally AND on Render (Linux)
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best_macro_model.weights.h5")
IMG_SIZE = 224

macro_model = None

if os.path.exists(WEIGHTS_PATH):
    print(f"Found weights at {WEIGHTS_PATH}. Rebuilding architecture...")
    try:
        _base = tf.keras.applications.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'  # Must match train.py initialization
        )

        # Match the exact freezing logic from train.py so the geometry of 
        # trainable vs non-trainable weights aligns perfectly
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
        # Build by running a dummy input so weights can load
        macro_model(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]))
        macro_model.load_weights(WEIGHTS_PATH, by_name=True)
        print("Model loaded successfully from weights!")
    except Exception as e:
        print(f"Warning: Could not load model weights: {e}. Using mock predictions.")
        macro_model = None
else:
    print("Warning: Trained model weights not found! Falling back to mock predictions.")


def predict_image(image_bytes):
    """
    Uses the EfficientNetB0 custom trained model to predict macros.
    """
    if macro_model is None:
        # Fallback: realistic average values per 100g of food
        # These will be overridden by Edamam if food_name is provided
        return [
            150.0,  # Calories (kcal per ~100g)
            100.0,  # Mass (g)
            5.0,    # Fat (g)
            20.0,   # Carbs (g)
            8.0     # Protein (g)
        ]
    
    # Process incoming image to match EfficientNetB0 input (224x224x3)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1
    
    # Predict
    # Output structure: [Calories, Mass, Fat, Carb, Protein]
    predictions = macro_model.predict(img_array)
    res = predictions[0]
    
    # Ensure no negative values (ReLU mostly handles this but just in case)
    return [max(0, float(val)) for val in res]

# --- [Built-in Nutrition Database — USDA per 100g values] ---
# Used as the primary source when user types a food name.
# Values are per 100g; we scale to actual serving size automatically.
NUTRITION_DB = {
    # eggs & dairy
    "egg":              {"calories": 155, "fat": 11.0, "carbs": 1.1,  "protein": 13.0},
    "boiled egg":       {"calories": 155, "fat": 11.0, "carbs": 1.1,  "protein": 13.0},
    "fried egg":        {"calories": 196, "fat": 15.0, "carbs": 0.8,  "protein": 14.0},
    "scrambled egg":    {"calories": 149, "fat": 9.5,  "carbs": 1.6,  "protein": 10.0},
    "milk":             {"calories": 61,  "fat": 3.3,  "carbs": 4.8,  "protein": 3.2},
    "cheese":           {"calories": 402, "fat": 33.0, "carbs": 1.3,  "protein": 25.0},
    "yogurt":           {"calories": 59,  "fat": 0.4,  "carbs": 3.6,  "protein": 10.0},
    # meat & fish
    "chicken breast":   {"calories": 165, "fat": 3.6,  "carbs": 0.0,  "protein": 31.0},
    "chicken":          {"calories": 165, "fat": 3.6,  "carbs": 0.0,  "protein": 31.0},
    "beef":             {"calories": 250, "fat": 15.0, "carbs": 0.0,  "protein": 26.0},
    "salmon":           {"calories": 208, "fat": 13.0, "carbs": 0.0,  "protein": 20.0},
    "tuna":             {"calories": 132, "fat": 1.0,  "carbs": 0.0,  "protein": 29.0},
    "shrimp":           {"calories": 99,  "fat": 1.7,  "carbs": 0.2,  "protein": 24.0},
    # grains & bread
    "rice":             {"calories": 130, "fat": 0.3,  "carbs": 28.0, "protein": 2.7},
    "white rice":       {"calories": 130, "fat": 0.3,  "carbs": 28.0, "protein": 2.7},
    "brown rice":       {"calories": 112, "fat": 0.9,  "carbs": 24.0, "protein": 2.6},
    "bread":            {"calories": 265, "fat": 3.2,  "carbs": 49.0, "protein": 9.0},
    "pasta":            {"calories": 131, "fat": 1.1,  "carbs": 25.0, "protein": 5.0},
    "oats":             {"calories": 389, "fat": 6.9,  "carbs": 66.0, "protein": 17.0},
    # vegetables
    "broccoli":         {"calories": 34,  "fat": 0.4,  "carbs": 7.0,  "protein": 2.8},
    "carrot":           {"calories": 41,  "fat": 0.2,  "carbs": 10.0, "protein": 0.9},
    "potato":           {"calories": 77,  "fat": 0.1,  "carbs": 17.0, "protein": 2.0},
    "salad":            {"calories": 15,  "fat": 0.2,  "carbs": 2.9,  "protein": 1.3},
    "spinach":          {"calories": 23,  "fat": 0.4,  "carbs": 3.6,  "protein": 2.9},
    # fruits
    "banana":           {"calories": 89,  "fat": 0.3,  "carbs": 23.0, "protein": 1.1},
    "apple":            {"calories": 52,  "fat": 0.2,  "carbs": 14.0, "protein": 0.3},
    "orange":           {"calories": 47,  "fat": 0.1,  "carbs": 12.0, "protein": 0.9},
    # other common foods
    "pizza":            {"calories": 266, "fat": 10.0, "carbs": 33.0, "protein": 11.0},
    "burger":           {"calories": 295, "fat": 14.0, "carbs": 24.0, "protein": 17.0},
    "sandwich":         {"calories": 218, "fat": 8.0,  "carbs": 28.0, "protein": 10.0},
    "soup":             {"calories": 50,  "fat": 1.5,  "carbs": 7.0,  "protein": 3.0},
    "sushi":            {"calories": 150, "fat": 0.6,  "carbs": 30.0, "protein": 6.0},
    "chocolate":        {"calories": 546, "fat": 31.0, "carbs": 60.0, "protein": 5.0},
    "peanut butter":    {"calories": 588, "fat": 50.0, "carbs": 20.0, "protein": 25.0},
    "avocado":          {"calories": 160, "fat": 15.0, "carbs": 9.0,  "protein": 2.0},
    "almonds":          {"calories": 579, "fat": 50.0, "carbs": 22.0, "protein": 21.0},
}

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
        # Partial match: find any entry whose key appears in the query or vice versa
        for key, values in NUTRITION_DB.items():
            if key in query_lower or query_lower in key:
                matched = values
                print(f"[LocalDB] Partial match: '{query_lower}' -> '{key}'")
                break

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

    # 3. Multimodal Food Recognition (Classification)
    detected_name = food_name
    detected_confidence = 0.0
    
    if not detected_name and classifier_model is not None:
        try:
            print("[Classifier] Running ImageNet Food Recognition...")
            img_clf = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_clf = img_clf.resize((224, 224))
            img_array_clf = tf.keras.utils.img_to_array(img_clf)
            img_array_clf = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_clf)
            img_array_clf = tf.expand_dims(img_array_clf, 0)
            
            # Predict top 3 classes
            preds = classifier_model.predict(img_array_clf)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
            
            # Use the highest confidence guess
            best_guess = decoded[0][1].replace("_", " ")
            detected_confidence = float(decoded[0][2])
            
            # Only trust it if confidence > 10%
            if detected_confidence > 0.10:
                detected_name = best_guess
                print(f"[Classifier] AI Recognition Guessed: '{detected_name}' ({detected_confidence*100:.1f}%)")
            else:
                print(f"[Classifier] AI Recognition was too uncertain. Top guess was: '{best_guess}' ({detected_confidence*100:.1f}%)")
        except Exception as e:
            print(f"[Classifier] Error running classification: {e}")

    # 4. Deep Learning: Run Regression Inference (Nutrition5k Model)
    try:
        prediction = predict_image(image_bytes)
        
        # Base model assumption: prediction[1] is the predicted Mass in grams
        predicted_mass_g = prediction[1]
        
        # Parse User's Manual Mass if provided
        calculated_mass_g = predicted_mass_g
        if serving_size is not None:
            if serving_unit == "oz":
                calculated_mass_g = serving_size * 28.3495
            elif serving_unit == "lbs":
                calculated_mass_g = serving_size * 453.592
            else: # Defaults to grams
                calculated_mass_g = serving_size

        # Scale AI macros linearly based on mass
        # Safeguard against zero division if the AI predicted 0 mass natively
        multiplier = 1.0
        if predicted_mass_g > 0:
            multiplier = calculated_mass_g / predicted_mass_g
            
        ai_macros = {
            "calories": prediction[0] * multiplier,
            "mass_grams": calculated_mass_g,
            "fat_grams": prediction[2] * multiplier,
            "carbs_grams": prediction[3] * multiplier,
            "protein_grams": prediction[4] * multiplier
        }
        
        print(f"AI Raw: {prediction}")
        print(f"AI Scaled: {ai_macros}")

        # 5. Multimodal NLP Override
        # If the user typed a food name (or the AI Classifier guessed one), query Edamam/Local DB
        final_macros = ai_macros
        data_source = "Deep Learning Vision"
        
        if detected_name and detected_name.strip() != "":
            nlp_macros = fetch_edamam_nutrition(food_query=detected_name.strip(), serving_size_g=calculated_mass_g)
            if nlp_macros:
                # Use Verified Database exclusively when food name is known (either typed or AI-detected)
                final_macros = {
                    "calories": nlp_macros["calories"],
                    "mass_grams": calculated_mass_g,
                    "fat_grams": nlp_macros["fat_grams"],
                    "carbs_grams": nlp_macros["carbs_grams"],
                    "protein_grams": nlp_macros["protein_grams"],
                }
                
                # Update data source description
                if food_name:
                    data_source = "NLP Database (Verified)"
                else:
                    data_source = f"AI Classifier + DB ({detected_confidence*100:.0f}% Conf)"

        return {
            "status": "success",
            "filename": file.filename,
            "detected_food_name": detected_name if detected_name else "Unknown (AI Vision Model)",
            "data_source_used": data_source,
            "nutrition_estimates": {
                "calories": round(final_macros["calories"], 1),
                "mass_grams": round(final_macros["mass_grams"], 1),
                "fat_grams": round(final_macros["fat_grams"], 1),
                "carbs_grams": round(final_macros["carbs_grams"], 1),
                "protein_grams": round(final_macros["protein_grams"], 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run the Cloud server
    uvicorn.run(app, host="0.0.0.0", port=8000)
