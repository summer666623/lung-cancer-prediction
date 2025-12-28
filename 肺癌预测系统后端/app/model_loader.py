import os
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "model2",
    "medical_risk_model.pkl"
)

print("🔍 正在加载模型：")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

# 🔥 加载的是 dict
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]

print("✅ 模型加载成功")


FEATURE_NAMES = [
    "Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy",
    "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease",
    "Balanced Diet", "Obesity", "Smoking", "Passive Smoker",
    "Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss",
    "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
    "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring"
]

def predict(features):
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    X_scaled = scaler.transform(df)
    return model.predict(X_scaled)

def predict_with_proba(features):
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    X_scaled = scaler.transform(df)

    pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    return int(pred[0]), proba[0]
