import requests

API_URL = "http://127.0.0.1:8000/predict"
LOCAL_IMAGE_PATH = r"Dataset\UECFOOD256\1\1.jpg"

print(f"[*] Testing API with local image: {LOCAL_IMAGE_PATH}...")

try:
    with open(LOCAL_IMAGE_PATH, "rb") as f:
        files = {"file": ("test_image.jpg", f, "image/jpeg")}
        print("[OK] Image loaded! Sending to API...\n")
        
        api_res = requests.post(API_URL, files=files, timeout=15)
        print(f"API Response Code: {api_res.status_code}\n")
        
        if api_res.status_code == 200:
            data = api_res.json()
            print("=== API RESULT ===")
            print(f"File: {data.get('filename')}")
            print(f"Detected Food: {data.get('detected_food_name')}")
            print(f"Data Source: {data.get('data_source_used')}")
            print("Nutrition:")
            for k, v in data.get('nutrition_estimates', {}).items():
                print(f"  - {k}: {v}")
        else:
            print("Error from API:", api_res.text)
except Exception as e:
    print("Failed to connect to API:", e)
