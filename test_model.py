import requests
import io
import sys
import os

FILES = {
    "pizza": "Dataset/UECFOOD256/18/11450.jpg",
    "raisin_bread": "Dataset/UECFOOD256/15/1400.jpg",
    "indian_food_1": "indian-food-6/test/images/03146376-1813-4436-a743-0a3900d39a3b_jpg.rf.78ff89b2d3b818aac6ed7bebd263194f.jpg",
    "indian_food_2": "indian-food-6/test/images/1709555268_shutterstock_1932825221_jpg.rf.8e2534c9bd717c9a96282af6aa416883.jpg"
}

API_URL = "http://127.0.0.1:8000/predict"

def test_image(name, path):
    print(f"\n--- Testing {name} ({path}) ---")
    if not os.path.exists(path):
        print("File not found.")
        return
        
    try:
        with open(path, 'rb') as f:
            files = {'file': (f'{name}.jpg', f.read(), 'image/jpeg')}
            
        resp = requests.post(API_URL, files=files, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Top Detection: {data.get('detected_food_name')}")
            print(f"Calories: {data.get('nutrition_estimates', {}).get('calories')}")
        else:
            print(f"Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Failed to test {name}: {e}")

if __name__ == "__main__":
    for name, path in FILES.items():
        test_image(name, path)
