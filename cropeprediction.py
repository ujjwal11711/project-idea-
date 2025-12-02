# =======================
# DIABETES PREDICTION + DIET RECOMMENDATION
# With Gradio Front-End (Run in Google Colab)
# =======================

# 📌 Install Libraries (Works in Colab)
!pip install gradio pandas scikit-learn

# 📌 Import Libraries
import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 📌 Load Dataset (Auto from URL)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# 📌 Split Data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Model
model = LogisticRegression(max_iter=1200)
model.fit(X_train, y_train)

# 📌 Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Trained 💉 Diabetes Accuracy:", round(accuracy * 100, 2), "%")

# ============================
# 🥗 Recommendation System
# ============================
def recommendations(pred, glucose, bmi):
    if pred == 1:  # Diabetes Positive
        rec = "⚠ High Diabetes Risk\n\n"

        # Diet advice
        if glucose > 130:
            rec += "🍽 Eat low sugar foods: Oats, Brown Rice, Green Vegetables, Nuts.\n"
        if bmi > 28:
            rec += "🏃‍♂ 30 min daily exercise required (Walking + Yoga).\n"
        rec += "🥗 Avoid: White rice, sweets, cold drinks, deep fried food."

    else:  # Negative
        rec = "✔ You are Safe from Diabetes 😄\n\n"
        rec += "🌿 Continue healthy lifestyle.\n"
        rec += "💧 Drink 2-3L water daily.\n🥗 Eat balanced diet + do light exercise."

    return rec

# ============================
# 🔮 Prediction Function
# ============================
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):

    user_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    pred = model.predict(user_data)[0]

    risk = "Diabetes Detected" if pred == 1 else "No Diabetes Risk"
    rec = recommendations(pred, Glucose, BMI)

    return f"🧪 Prediction: {risk}\n\n📌 Suggestions:\n{rec}"

# ============================
# 💻 GRADIO UI
# ============================
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose Level"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ],
    outputs="text",
    title="🩺 Diabetes Prediction + Diet Planner",
    description="Enter medical values to predict diabetes risk and get diet recommendations."
)

interface.launch(debug=True)
