import os
import csv as _csv
import time
import logging
import jwt
import requests
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Depends, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()


# In a real environment, load this safely from an environment variable!
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "my_super_secret_jwt_key")
ALGORITHM = "HS256"

# ----------------------------------------
# Security Audit Logging
# ----------------------------------------
# A centralized security audit log captures all critical events.
# This supports incident response, forensic investigation, and
# compliance requirements (ISO 27001, NIST, DPDP Act 2023).

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

audit_logger = logging.getLogger("security_audit")
audit_logger.setLevel(logging.INFO)
audit_handler = logging.FileHandler(os.path.join(LOG_DIR, "security_audit.log"), encoding="utf-8")
audit_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
audit_logger.addHandler(audit_handler)

def log_security_event(event_type: str, details: str, ip: str = "unknown", severity: str = "INFO"):
    """Log a security-relevant event for forensic analysis."""
    level = getattr(logging, severity.upper(), logging.INFO)
    audit_logger.log(level, f"[{event_type}] IP={ip} | {details}")

# Edamam Nutrition API Keys
EDAMAM_APP_ID = "cc3fe44a"
EDAMAM_APP_KEY = "6be30bfa2e95a94ee8997ef0d195a6ec"

# 1. Initialize FastAPI & Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Food Nutrition Estimator API")
app.state.limiter = limiter
app.add_exception_handler(429, lambda request, exc: JSONResponse(
    status_code=429, content={"detail": "Too many requests. Please slow down."}
))

# ----------------------------------------
# Security Headers Middleware
# ----------------------------------------
# Injects industry-standard HTTP security headers into every response.
# Protects against Clickjacking (X-Frame-Options), MIME sniffing
# (X-Content-Type-Options), XSS (CSP), and enforces HTTPS (HSTS).

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds strict security headers to every HTTP response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"            # Prevent MIME-type sniffing
        response.headers["X-Frame-Options"] = "DENY"                      # Prevent Clickjacking
        response.headers["X-XSS-Protection"] = "1; mode=block"            # XSS filter (legacy browsers)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"  # Force HTTPS
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"  # Control referrer leakage
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' https://*.onrender.com https://*.hf.space http://127.0.0.1:* http://localhost:*;"
        )
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"  # Restrict browser APIs
        return response

# 2. Add Security Middleware
app.add_middleware(SecurityHeadersMiddleware)  # CyberSecurity: Strict HTTP headers
app.add_middleware(SlowAPIMiddleware) # CyberSecurity: Prevents DDoS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://YOUR-FRONTEND-APP-NAME.netlify.app", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- [YOLOv8 Custom Food Detection Models] ---
# We load BOTH the baseline model (256 classes) and the Indian Food model
# so it can recognize foods from both datasets simultaneously.

BASE_YOLO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "detect", "food_detector5", "weights", "best.pt")
INDIAN_YOLO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "detect", "indian_food_v2", "weights", "best.pt")
FALLBACK_YOLO = "yolov8n.pt"

yolo_models = []
try:
    from ultralytics import YOLO
    
    if os.path.exists(BASE_YOLO_PATH):
        print("[YOLO] Loading BASE trained food model (256 classes)...")
        yolo_models.append(YOLO(BASE_YOLO_PATH))
        
    if os.path.exists(INDIAN_YOLO_PATH):
        print("[YOLO] Loading INDIAN trained food model...")
        yolo_models.append(YOLO(INDIAN_YOLO_PATH))

    if not yolo_models:
        print(f"[YOLO] Custom models not found. Loading generic {FALLBACK_YOLO}...")
        yolo_models.append(YOLO(FALLBACK_YOLO))
        print("[YOLO] Generic model loaded.")
        
    USE_YOLO = True
except ImportError:
    print("[YOLO] ultralytics not installed. Multi-food detection disabled.")
    USE_YOLO = False
except Exception as e:
    print(f"[YOLO] Error loading YOLO models: {e}")
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

    # Manual overrides / items not in Nutrition5k (imported from external file)
    from backend.nutrition_data import MANUAL_NUTRITION_DB
    manual = MANUAL_NUTRITION_DB
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
        # Partial match: Try to find the longest key in NUTRITION_DB that is a substring of the query
        # Sorting keys by length descending ensures we match "pork chop" before "pork"
        longest_match_key = None
        for key in sorted(NUTRITION_DB.keys(), key=len, reverse=True):
            if len(key) >= 3 and (key in query_lower):
                longest_match_key = key
                break
                
        if not longest_match_key:
            # Fallback for when query is short but matches part of a longer DB key
            for key in sorted(NUTRITION_DB.keys(), key=len, reverse=True):
                if len(query_lower) >= 3 and (query_lower in key):
                    longest_match_key = key
                    break

        if longest_match_key:
            matched = NUTRITION_DB[longest_match_key]
            print(f"[LocalDB] Partial match: '{query_lower}' -> '{longest_match_key}'")
        
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
            # TEST ONLY: Dummy Micronutrients
            "micronutrients": {
                "vitamin_a_iu": round(150.0 * scale, 1),
                "vitamin_c_mg": round(12.5 * scale, 1),
                "iron_mg": round(2.1 * scale, 1),
                "calcium_mg": round(45.0 * scale, 1)
            }
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
                    "protein_grams": round(nutrients.get("PROCNT", {}).get("quantity", 0), 1),
                    # TEST ONLY: Fallback API Micronutrients
                    "micronutrients": {
                        "vitamin_a_iu": round(nutrients.get("VITA_IU", {}).get("quantity", 50.0), 1),
                        "vitamin_c_mg": round(nutrients.get("VITC", {}).get("quantity", 3.0), 1),
                        "iron_mg": round(nutrients.get("FE", {}).get("quantity", 1.2), 1),
                        "calcium_mg": round(nutrients.get("CA", {}).get("quantity", 20.0), 1)
                    }
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
# 3. Security: JWT Authentication & RBAC
# ----------------------------------------
# JWT tokens carry a 'role' claim (e.g., "user" or "admin").
# This implements the RBAC access control model where permissions
# are determined by the user's assigned role, not individual identity.

def verify_token(req: Request):
    """
    CyberSecurity Requirement:
    Ensure the user is authorized before they can upload images.
    Returns the decoded JWT payload including role information.
    """
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        ip = req.client.host if req.client else "unknown"
        log_security_event("AUTH_FAILURE", "Missing or invalid Authorization header", ip=ip, severity="WARNING")
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        ip = req.client.host if req.client else "unknown"
        log_security_event("AUTH_FAILURE", "Expired JWT token presented", ip=ip, severity="WARNING")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        ip = req.client.host if req.client else "unknown"
        log_security_event("AUTH_FAILURE", "Invalid/tampered JWT token presented", ip=ip, severity="CRITICAL")
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(required_role: str, payload: dict):
    """
    RBAC enforcement: Checks that the JWT token contains
    the required role. If not, the request is denied (403 Forbidden).
    """
    user_role = payload.get("role", "user")
    if user_role != required_role:
        log_security_event("RBAC_DENIED", f"User '{payload.get('user_id')}' with role '{user_role}' attempted to access '{required_role}'-only resource", severity="WARNING")
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. This endpoint requires '{required_role}' role. Your role: '{user_role}'."
        )


# ----------------------------------------
# 4. API Endpoints
# ----------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Food Nutrition API is running"}

@app.post("/login")
@limiter.limit("10/minute") # Strict limit on login endpoints
def login(request: Request, role: Optional[str] = Form("user")):
    """
    Authenticate and receive a JWT access token.
    Accepts a 'role' parameter ("user" or "admin") for RBAC enforcement.
    """
    ip = request.client.host if request.client else "unknown"
    # Validate role
    allowed_roles = ["user", "admin"]
    if role not in allowed_roles:
        log_security_event("LOGIN_FAILURE", f"Invalid role '{role}' requested", ip=ip, severity="WARNING")
        raise HTTPException(status_code=400, detail=f"Invalid role. Allowed: {allowed_roles}")
    
    expiration = time.time() + 3600  # 1 hour TTL
    token = jwt.encode(
        {"user_id": "nutrivision_user", "role": role, "exp": expiration},
        SECRET_KEY, algorithm=ALGORITHM
    )
    log_security_event("LOGIN_SUCCESS", f"User authenticated with role='{role}'", ip=ip)
    return {"access_token": token, "token_type": "bearer", "role": role}


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
    ip = request.client.host if request.client else "unknown"
    log_security_event("PREDICT_REQUEST", f"Image upload: '{file.filename}' ({file.content_type})", ip=ip)
    # 1. CyberSecurity Check: Validate File Type
    if not file.content_type.startswith("image/"):
        log_security_event("INVALID_UPLOAD", f"Rejected non-image file: '{file.filename}' (type={file.content_type})", ip=ip, severity="WARNING")
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")

    # 2. Read Image Data
    image_bytes = await file.read()
    
    # CyberSecurity Check: Validate File Size (Max 5MB)
    if len(image_bytes) > 5 * 1024 * 1024:
        log_security_event("OVERSIZED_UPLOAD", f"Rejected oversized file: '{file.filename}' ({len(image_bytes)} bytes)", ip=ip, severity="WARNING")
        raise HTTPException(status_code=400, detail="File too large. Max 5MB allowed.")

    # 3. Object Detection (YOLOv8)
    try:
        detected_foods = []
        
        # Check if the user bypassed AI via Manual Text Input
        if food_name and food_name.strip() != "":
            detected_foods.append(food_name.strip())
            print(f"[YOLO] User override: {food_name}")
        elif USE_YOLO and len(yolo_models) > 0:
            try:
                print(f"[YOLO] Scanning using {len(yolo_models)} models...")
                img_clf = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Resize very large images to 640px (YOLO's native resolution)
                # This improves detection accuracy and reduces processing time
                max_dim = max(img_clf.size)
                if max_dim > 1280:
                    scale = 1280 / max_dim
                    new_size = (int(img_clf.size[0] * scale), int(img_clf.size[1] * scale))
                    img_clf = img_clf.resize(new_size, Image.LANCZOS)
                    print(f"[YOLO] Resized image to {new_size} for optimal detection")

                all_detections = [] # To track tuples of (class_name, confidence)
                
                for model in yolo_models:
                    print(f"[YOLO] Running model with {len(model.names)} classes...")
                    
                    # === Tiered Confidence Strategy ===
                    # Tier 1: High confidence (conf=0.30) — reliable detections
                    # Tier 2: Lower confidence (conf=0.15) — catches harder cases
                    # Tier 3: Very low confidence (conf=0.10) with flipped image — last resort
                    
                    confidence_tiers = [
                        (0.30, img_clf, "Tier 1 (conf=0.30)"),
                        (0.15, img_clf, "Tier 2 (conf=0.15)"),
                    ]
                    
                    model_found_food = False
                    for conf_threshold, img_input, tier_label in confidence_tiers:
                        results = model(img_input, conf=conf_threshold, iou=0.45, verbose=False)
                        
                        for box in results[0].boxes:
                            cls_name = model.names[int(box.cls)].replace("_", " ")
                            conf_val = float(box.conf)
                            
                            # Calibration heuristic: Smaller models (like the 29-class one) 
                            # easily get overconfident on unknown shapes. We boost the large 
                            # base model's score so it doesn't get overridden by false positives.
                            adjusted_conf = conf_val + 0.30 if len(model.names) > 100 else conf_val
                            
                            all_detections.append((cls_name, adjusted_conf, conf_val))
                            model_found_food = True
                            print(f"[YOLO] {tier_label}: {cls_name} (conf={conf_val:.3f}, adjusted={adjusted_conf:.3f})")
                        
                        if model_found_food:
                            break  # Stop trying lower tiers for this specific model if found something
                    
                    # Tier 3: Try with horizontally flipped image (different perspective)
                    if not model_found_food:
                        from PIL import ImageOps
                        img_flipped = ImageOps.mirror(img_clf)
                        results_flip = model(img_flipped, conf=0.10, iou=0.45, verbose=False)
                        for box in results_flip[0].boxes:
                            cls_name = model.names[int(box.cls)].replace("_", " ")
                            conf_val = float(box.conf)
                            
                            adjusted_conf = conf_val + 0.30 if len(model.names) > 100 else conf_val
                            
                            all_detections.append((cls_name, adjusted_conf, conf_val))
                            print(f"[YOLO] Tier 3 (flipped, conf=0.10): {cls_name} (conf={conf_val:.3f}, adjusted={adjusted_conf:.3f})")
                
                if all_detections:
                    # Sort all detections by the ADJUSTED confidence score descending
                    all_detections.sort(key=lambda x: x[1], reverse=True)
                    # Pick only the highest confidence detection
                    best_food = all_detections[0][0]
                    detected_foods = [best_food]
                    print(f"[YOLO] Top confidence detection selected: {detected_foods} (raw_conf={all_detections[0][2]:.3f})")
                else:
                    detected_foods = []
                    print("[YOLO] No food objects detected after all models and tiers.")
            except Exception as e:
                print(f"[YOLO] Error running object detection: {e}")

        # 4. Nutrition Calculation Loop
        # We will sum the macros for ALL detected foods!
        total_macros = {
            "calories": 0.0,
            "mass_grams": 0.0, 
            "fat_grams": 0.0,
            "carbs_grams": 0.0,
            "protein_grams": 0.0,
            "micronutrients": {
                "vitamin_a_iu": 0.0,
                "vitamin_c_mg": 0.0,
                "iron_mg": 0.0,
                "calcium_mg": 0.0
            }
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
            # No food detected — return a clear error instead of fake data
            print("[YOLO] No food detected in the image.")
            raise HTTPException(
                status_code=400,
                detail="Could not identify any food in this image. Please try uploading a clearer photo of a food item."
            )
        else:
            data_source_used = "YOLOv8 + Nutrition DB"
            total_mass_g = 0.0
            failed_foods = []
            
            for food in detected_foods:
                # 1. Try local DB first. 2. Fall back to Edamam API if not found.
                item_macros = fetch_edamam_nutrition(food_query=food, serving_size_g=per_item_mass_g)
                
                if item_macros:
                    total_macros["calories"] += item_macros["calories"]
                    total_macros["fat_grams"] += item_macros["fat_grams"]
                    total_macros["carbs_grams"] += item_macros["carbs_grams"]
                    total_macros["protein_grams"] += item_macros["protein_grams"]
                    total_mass_g += item_macros.get("mass_grams", per_item_mass_g)
                    
                    if "micronutrients" in item_macros:
                        total_macros["micronutrients"]["vitamin_a_iu"] += item_macros["micronutrients"].get("vitamin_a_iu", 0)
                        total_macros["micronutrients"]["vitamin_c_mg"] += item_macros["micronutrients"].get("vitamin_c_mg", 0)
                        total_macros["micronutrients"]["iron_mg"] += item_macros["micronutrients"].get("iron_mg", 0)
                        total_macros["micronutrients"]["calcium_mg"] += item_macros["micronutrients"].get("calcium_mg", 0)
                else:
                    failed_foods.append(food)
                    print(f"[YOLO] Warning: {food} detected but not found in DB.")
            
            total_macros["mass_grams"] = total_mass_g
            
            # If ALL lookups failed, return a clear error with the detected food names
            if total_macros["calories"] < 1:
                 food_names = ", ".join([f.capitalize() for f in detected_foods])
                 print(f"[YOLO] All lookups failed for: {food_names}")
                 raise HTTPException(
                    status_code=400,
                    detail=f"Detected '{food_names}' but nutrition data is not available for this item. Try a different food image."
                 )

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
                "protein_grams": round(total_macros["protein_grams"], 1),
                "micronutrients": {
                    "vitamin_a_iu": round(total_macros["micronutrients"]["vitamin_a_iu"], 1),
                    "vitamin_c_mg": round(total_macros["micronutrients"]["vitamin_c_mg"], 1),
                    "iron_mg": round(total_macros["micronutrients"]["iron_mg"], 1),
                    "calcium_mg": round(total_macros["micronutrients"]["calcium_mg"], 1)
                }
            }
        }
    except HTTPException:
        # Re-raise HTTP exceptions (e.g. 400 no-food-detected) without converting to 500
        raise
    except Exception as e:
        log_security_event("PREDICT_ERROR", f"Processing failed: {str(e)}", ip=ip, severity="ERROR")
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")


# ----------------------------------------
# Admin-Only Endpoints (RBAC Protected)
# ----------------------------------------
# Only users with role="admin" in their JWT can access these endpoints.

@app.get("/admin/logs")
@limiter.limit("10/minute")
def get_audit_logs(request: Request, user: dict = Depends(verify_token)):
    """
    RBAC-Protected Endpoint: View security audit logs.
    Requires: JWT token with role='admin'.
    Regular 'user' role tokens will be rejected with 403 Forbidden.
    """
    require_role("admin", user)
    
    log_path = os.path.join(LOG_DIR, "security_audit.log")
    if not os.path.exists(log_path):
        return {"logs": [], "message": "No audit logs found."}
    
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-100:]  # Last 100 entries
    
    log_security_event("ADMIN_ACCESS", f"Admin '{user.get('user_id')}' viewed audit logs", 
                       ip=request.client.host if request.client else "unknown")
    return {
        "logs": [line.strip() for line in lines],
        "total_entries": len(lines),
        "accessed_by": user.get("user_id")
    }


@app.get("/admin/system-health")
@limiter.limit("10/minute")
def system_health(request: Request, user: dict = Depends(verify_token)):
    """
    RBAC-Protected Endpoint: View system health dashboard.
    Requires: JWT token with role='admin'.
    """
    require_role("admin", user)
    
    log_security_event("ADMIN_ACCESS", f"Admin '{user.get('user_id')}' viewed system health",
                       ip=request.client.host if request.client else "unknown")
    return {
        "status": "healthy",
        "yolo_model_loaded": USE_YOLO,
        "yolo_models_count": len(yolo_models) if USE_YOLO else 0,
        "nutrition_db_entries": len(NUTRITION_DB),
        "log_directory": LOG_DIR,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ----------------------------------------
# Privacy Policy & Legal Compliance
# ----------------------------------------
# Serves a compliance page referencing India's DPDP Act 2023 and
# the EU's GDPR. Documents the data minimization strategy employed
# by NutriVision AI (images processed in-memory, never stored).

@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    """Serves the Privacy Policy & DPDP Act / GDPR compliance page."""
    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "privacy_policy.html"))
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = "<h1>Privacy Policy</h1><p>Document not found.</p>"
    return HTMLResponse(content=content)


if __name__ == "__main__":
    import uvicorn
    log_security_event("SERVER_START", "NutriVision API server starting", severity="INFO")
    # Run the Cloud server
    uvicorn.run(app, host="0.0.0.0", port=8000)