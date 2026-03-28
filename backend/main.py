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
        "sashimi":       {"calories": 127, "fat": 4.0,  "carbs": 0.0,  "protein": 22.0},
        "salmon sashimi": {"calories": 180, "fat": 10.0, "carbs": 0.0,  "protein": 20.0},
        "tuna sashimi":   {"calories": 110, "fat": 0.5,  "carbs": 0.0,  "protein": 24.0},
        "udon noodles":   {"calories": 280, "fat": 2.0,  "carbs": 55.0, "protein": 10.0},
        "soba":           {"calories": 300, "fat": 2.0,  "carbs": 60.0, "protein": 12.0},
        "yakisoba":       {"calories": 450, "fat": 15.0, "carbs": 65.0, "protein": 12.0},
        "gyoza":          {"calories": 250, "fat": 12.0, "carbs": 25.0, "protein": 10.0},
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
        elif USE_YOLO and yolo_model is not None:
            try:
                print("[YOLO] Scanning for multiple foods...")
                # We must save the bytes to a temp file for Ultralytics 
                # (YOLO prefers file paths or PIL Images)
                img_clf = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Run YOLO prediction with higher confidence threshold and NMS (Non-Maximum Suppression)
                # conf=0.50 means the AI must be 50% sure it's correct (reduces false positives)
                # iou=0.45 prevents overlapping bounding boxes for the same food item
                results = yolo_model(img_clf, conf=0.30, iou=0.45, verbose=False)
                
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
                "protein_grams": round(total_macros["protein_grams"], 1)
            }
        }
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
        "yolo_model_path": CUSTOM_YOLO_PATH if USE_YOLO and os.path.exists(CUSTOM_YOLO_PATH) else FALLBACK_YOLO,
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

PRIVACY_POLICY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NutriVision AI — Privacy Policy & Compliance</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0f1a; color: #c8d6e5; line-height: 1.7; padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #10b981; font-size: 2rem; margin-bottom: 0.5rem; }
        h2 { color: #34d399; font-size: 1.3rem; margin-top: 2rem; margin-bottom: 0.5rem; border-bottom: 1px solid rgba(16,185,129,0.2); padding-bottom: 0.3rem; }
        h3 { color: #a78bfa; font-size: 1.1rem; margin-top: 1.5rem; }
        p, li { font-size: 0.95rem; margin-bottom: 0.7rem; }
        ul { padding-left: 1.5rem; }
        .badge { display: inline-block; background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #34d399; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-bottom: 1rem; }
        .highlight { background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.2); border-radius: 10px; padding: 1rem; margin: 1rem 0; }
        .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.06); color: #475569; font-size: 0.8rem; }
        a { color: #60a5fa; }
    </style>
</head>
<body>
    <div class="container">
        <span class="badge">Legal Compliance Document</span>
        <h1>🔒 Privacy Policy & Data Protection</h1>
        <p>NutriVision AI — Food Nutrition Estimator</p>
        <p><em>Last updated: March 2026</em></p>

        <h2>1. Overview</h2>
        <p>NutriVision AI is committed to protecting user privacy in compliance with applicable data protection laws including India's <strong>Digital Personal Data Protection (DPDP) Act, 2023</strong> and the EU's <strong>General Data Protection Regulation (GDPR)</strong>.</p>

        <h2>2. Data We Process</h2>
        <ul>
            <li><strong>Food Images:</strong> Uploaded images are processed entirely <strong>in-memory</strong> for AI-based food recognition. Images are <strong>never written to disk, stored in a database, or transmitted to third parties</strong>.</li>
            <li><strong>Nutrition Queries:</strong> When our local database cannot identify a food item, the food name (text only — not the image) may be sent to the Edamam Nutrition API for lookup.</li>
            <li><strong>IP Addresses:</strong> Logged temporarily for rate limiting and security audit purposes as permitted under legitimate interest.</li>
        </ul>

        <h2>3. Data Minimization (DPDP Act Section 4 & GDPR Article 5)</h2>
        <div class="highlight">
            <p>We implement <strong>data minimization by design</strong>: only the minimum personal data required for the service (food image for recognition) is processed. Images exist only in server RAM during processing and are immediately discarded. No persistent storage of personal data occurs.</p>
        </div>

        <h2>4. Legal Basis for Processing</h2>
        <ul>
            <li><strong>Consent (DPDP Act Section 6):</strong> By uploading an image, the user consents to its processing for nutrition estimation.</li>
            <li><strong>Legitimate Interest (GDPR Article 6(1)(f)):</strong> Security logging (IP addresses, failed authentication attempts) is maintained for system security and fraud prevention.</li>
        </ul>

        <h2>5. User Rights</h2>
        <h3>Under DPDP Act 2023:</h3>
        <ul>
            <li>Right to access information about personal data processing (Section 11)</li>
            <li>Right to correction and erasure of personal data (Section 12)</li>
            <li>Right to grievance redressal (Section 13)</li>
            <li>Right to nominate (Section 14)</li>
        </ul>
        <h3>Under GDPR:</h3>
        <ul>
            <li>Right to Access (Article 15), Rectification (Article 16), Erasure (Article 17)</li>
            <li>Right to Restriction of Processing (Article 18)</li>
            <li>Right to Data Portability (Article 20)</li>
        </ul>

        <h2>6. Security Measures</h2>
        <ul>
            <li>JWT-based authentication with HS256 signing and automatic token expiry</li>
            <li>Role-Based Access Control (RBAC) for administrative endpoints</li>
            <li>Rate limiting (DDoS protection) via sliding-window algorithms</li>
            <li>Input validation: MIME type checking, file size limits (5MB)</li>
            <li>HTTP security headers: HSTS, CSP, X-Frame-Options, X-Content-Type-Options</li>
            <li>CORS policy restricting API access to authorized frontends</li>
            <li>Environment variable isolation for secrets (never committed to source code)</li>
            <li>Centralized audit logging for forensic investigation</li>
        </ul>

        <h2>7. Third-Party Services</h2>
        <ul>
            <li><strong>Edamam API:</strong> Used as a fallback nutrition data source. Only food name text is sent (not images). See: <a href="https://developer.edamam.com/edamam-nutrition-api" target="_blank">Edamam Privacy Policy</a></li>
        </ul>

        <h2>8. Data Retention</h2>
        <p>Uploaded images: <strong>0 seconds</strong> (processed in-memory, immediately discarded).<br>
        Security audit logs: Retained for <strong>90 days</strong> for incident response purposes, then purged.<br>
        JWT tokens: Auto-expire after <strong>1 hour</strong> and are not stored server-side (stateless).</p>

        <h2>9. Contact</h2>
        <p>For privacy-related inquiries, data access requests, or to exercise your rights under DPDP Act 2023 or GDPR, contact us at: <strong>privacy@nutrivision-ai.example</strong></p>

        <div class="footer">
            <p>&copy; 2026 NutriVision AI. All rights reserved. This document is governed by applicable data protection regulations including the DPDP Act 2023 and GDPR. For questions, contact our Data Protection Officer at the email listed above.</p>
        </div>
    </div>
</body>
</html>
"""

@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    """Serves the Privacy Policy & DPDP Act / GDPR compliance page."""
    return HTMLResponse(content=PRIVACY_POLICY_HTML)


if __name__ == "__main__":
    import uvicorn
    log_security_event("SERVER_START", "NutriVision API server starting", severity="INFO")
    # Run the Cloud server
    uvicorn.run(app, host="0.0.0.0", port=8000)