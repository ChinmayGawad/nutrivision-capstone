# NutriVision AI 🥗

NutriVision AI is a smart food-recognition and nutrition-estimation web application. Using a custom-trained **YOLOv8** object detection model, the app can identify up to 256 different food classes from a single image and calculate detailed macro-nutritional breakdowns (Calories, Protein, Carbs, Fat). 

It features a fallback integration with the **Edamam Nutrition API** to accurately estimate macros for complex, multi-word dishes that aren't available in standard offline databases.

## 🚀 Features
- **Multi-Food Detection:** Identifies multiple distinct food items on a single plate simultaneously using YOLOv8.
- **Smart Nutrition Estimation:** Calculates base nutrition using a bundled 500+ item offline database.
- **API Fallback:** Automatically queries Edamam for complex queries (e.g., "chicken curry", "beef bowl") to ensure accurate macros.
- **Sleek UI:** Modern, responsive dark-mode frontend with Chart.js visualization.

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, Uvicorn
- **AI/ML:** Ultralytics YOLOv8 (trained on UECFOOD256)
- **Frontend:** HTML, Vanilla JS, Tailwind CSS, Chart.js

## 📦 Local Setup Instructions

### 1. Requirements
- Python 3.9+
- A trained YOLOv8 model file (`best.pt`) placed in the project root. (You can train one yourself using the provided `training/train_cloud.ipynb` on Kaggle!).

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/food-recognition-project.git
cd food-recognition-project
python -m venv venv
venv\Scripts\activate   # (On Windows) or `source venv/bin/activate` (On Mac/Linux)
pip install -r requirements.txt
```

### 3. Environment Variables (Optional but recommended)
To get the most accurate nutrition results for complex dishes, set up an account at Edamam (Developer API) and create a `.env` file in the project root:
```env
EDAMAM_APP_ID=your_id_here
EDAMAM_APP_KEY=your_key_here
```

### 4. Running the App
Start the FastAPI server:
```bash
uvicorn backend.main:app --reload
```
Open `frontend/index.html` in any modern web browser to use the interface!
