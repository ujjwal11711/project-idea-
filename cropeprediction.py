!pip install -q gradio pandas scikit-learn

import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------
# 📌 Load Dataset (WORKING URL)
# ----------------------------------------------------------
url = "https://raw.githubusercontent.com/arzzahid66/Optimizing_Agricultural_Production/master/Crop_recommendation.csv"
df = pd.read_csv(url)

print("Dataset Loaded Successfully ✅")
print(df.head())

# ----------------------------------------------------------
# 📌 Split Data
# ----------------------------------------------------------
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 📌 Train Model
# ----------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Model Trained Successfully 🌾")

# ----------------------------------------------------------
# 📌 Fertilizer Suggestion Logic
# ----------------------------------------------------------
fertilizer_data = {
    "rice": "Urea, DAP, Potash — nitrogen-rich fertilizer chahiye.",
    "wheat": "NPK 2:1:1 + Urea se accha yield milega.",
    "maize": "NPK 3:2:1, saath me organic compost / gobar khaad.",
    "sugarcane": "Ammonium sulphate + potash, zyda pani bhi chahiye.",
    "cotton": "Phosphorus rich fertilizer + organic manure.",
    "banana": "High Potassium (K) — MOP + compost.",
    "apple": "Calcium nitrate + organic khaad.",
    "coffee": "NPK 15:15:15 + regular compost."
}

def fertilizer_suggestion(crop):
    crop = crop.lower()
    return fertilizer_data.get(
        crop,
        "General advice: NPK balanced fertilizer + organic compost use karo."
    )

# ----------------------------------------------------------
# 📌 Prediction Function
# ----------------------------------------------------------
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):

    # Input ko model ke format me daalna
    user_input = [[N, P, K, temperature, humidity, ph, rainfall]]

    predicted_crop = model.predict(user_input)[0]
    fertilizer = fertilizer_suggestion(predicted_crop)

    output = f"""
🌾 *Recommended Crop:* {predicted_crop.capitalize()}

🧪 *Fertilizer Suggestion:*
{fertilizer}

📌 *Your Soil Values:*
- Nitrogen (N): {N}
- Phosphorus (P): {P}
- Potassium (K): {K}
- Temperature: {temperature} °C
- Humidity: {humidity} %
- Soil pH: {ph}
- Rainfall: {rainfall} mm
"""
    return output

# ----------------------------------------------------------
# 💻 GRADIO UI
# ----------------------------------------------------------
interface = gr.Interface(
    fn=recommend_crop,
    inputs=[
        gr.Number(label="N (Nitrogen level)"),
        gr.Number(label="P (Phosphorus level)"),
        gr.Number(label="K (Potassium level)"),
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Humidity (%)"),
        gr.Number(label="Soil pH"),
        gr.Number(label="Rainfall (mm)"),
    ],
    outputs="markdown",
    title="🌱 Crop Recommendation + Fertilizer Suggestion",
    description="Enter soil values to get the best crop recommendation and fertilizer advice."
)

# Colab me web app kholne ke liye
interface.launch(share=True, debug=True)
