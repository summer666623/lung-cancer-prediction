import sys
from flask import request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from flask import Blueprint, request, jsonify
from app.model_loader import predict_with_proba
from models.model2_incidence_predictor import LungCancerIncidenceService



incidence_service = LungCancerIncidenceService()
api_bp = Blueprint("api", __name__)

# ===============================
# 健康检查
# ===============================
@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ===============================
# 环境风险预测（唯一已实现模型）
# ===============================
@api_bp.route("/predict", methods=["POST"])
def predict_risk():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "请求体不是合法 JSON"}), 400

    # ⚠️ 模型训练时的特征顺序（必须严格一致）
    feature_order = [
        "Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy",
        "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease",
        "Balanced Diet", "Obesity", "Smoking", "Passive Smoker",
        "Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss",
        "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
        "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring"
    ]

    # ===============================
    # 1️⃣ 强校验特征是否缺失
    # ===============================
    missing = [k for k in feature_order if k not in data]
    if missing:
        return jsonify({
            "error": "缺少必要特征",
            "missing_features": missing
        }), 400

    # ===============================
    # 2️⃣ 构造模型输入
    # ===============================
    try:
        features = [[int(data[k]) for k in feature_order]]
    except Exception as e:
        return jsonify({
            "error": "特征解析失败",
            "detail": str(e)
        }), 400

    # ===============================
    # 3️⃣ 调用模型
    # ===============================
    try:
        risk_code, proba = predict_with_proba(features)

        distribution = {
            "low": float(proba[0]),
            "medium": float(proba[1]),
            "high": float(proba[2]),
        }

        risk_level_map = {0: "低风险", 1: "中风险", 2: "高风险"}
        risk_level = risk_level_map.get(risk_code, "未知")

    except Exception as e:
        return jsonify({
            "error": "模型预测失败",
            "detail": str(e)
        }), 500

    risk_level_map = {0: "低风险", 1: "中风险", 2: "高风险"}
    risk_level = risk_level_map.get(risk_code, "未知")

    return jsonify({
        "risk_code": risk_code,
        "risk_level": risk_level,
        "distribution": distribution  # ← 用模型真实概率
    })

@api_bp.route("/predict/incidence", methods=["POST"])
def predict_incidence():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "请求体不是合法 JSON"}), 400

    # ✅ 前端字段 → 模型字段 映射
    field_map = {
        "age": "年龄",
        "smoking_years": "吸烟年限",
        "pack_per_day": "日吸烟量",
        "family_history": "家族病史",
        "air_pollution_index": "空气污染指数",
        "chronic_lung_disease": "慢性肺病",
        "lung_cancer_prob": "癌症发病率"
    }

    try:
        mapped_data = {}
        for front_key, model_key in field_map.items():
            if front_key not in data:
                raise ValueError(f"缺少字段: {front_key}")
            mapped_data[model_key] = data[front_key]

        probability = incidence_service.predict_incidence_probability(mapped_data)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({
            "error": "模型预测失败",
            "detail": str(e)
        }), 500

    return jsonify({
        "incidence_probability": probability
    })


MODEL_DIR = r"D:\项目\肺癌预测系统后端\app\model2"
cox_model = joblib.load(os.path.join(MODEL_DIR, "cox_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_survival_model.pkl"))

# Cox + XGBoost 所需特征
SURVIVAL_FEATURES = [
    "race", "insurance", "family_history", "diabetes", "hypertension",
    "heart_disease", "copd", "kidney_disease", "autoimmune",
    "performance_score", "bp_stage", "other_comorbidity", "total_comorbidity",
    "perf_copd", "perf_kidney"
]


# =========================
# 2️⃣ 新增路由
# =========================
# Flask 后端 survival 路由
MODEL_DIR = r"D:\项目\肺癌预测系统后端\app\model2"
cox_model = joblib.load(os.path.join(MODEL_DIR, "cox_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_survival_model.pkl"))
# 前端字段 -> 模型字段映射
SURVIVAL_FIELD_MAP = {
    "Ethnicity": "race",
    "Insurance_Type": "insurance",
    "Family_History": "family_history",
    "Comorbidity_Diabetes": "diabetes",
    "Comorbidity_Hypertension": "hypertension",
    "Comorbidity_Heart_Disease": "heart_disease",
    "Comorbidity_Chronic_Lung_Disease": "copd",
    "Comorbidity_Kidney_Disease": "kidney_disease",
    "Comorbidity_Autoimmune_Disease": "autoimmune",
    "Comorbidity_Other": "other_comorbidity",
    "Performance_Status": "performance_score",
    "Blood_Pressure": "bp_stage"
}

@api_bp.route("/predict/survival", methods=["POST"])
def predict_survival():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "请求体不是合法 JSON"}), 400

    # 前端字段 → 模型字段映射
    mapped_data = {}
    for front_key, model_key in SURVIVAL_FIELD_MAP.items():
        if front_key not in data:
            return jsonify({"error": f"缺少字段: {front_key}"}), 400
        mapped_data[model_key] = data[front_key]

    # 计算衍生特征
    mapped_data["perf_copd"] = mapped_data["performance_score"] * mapped_data.get("copd", 0)
    mapped_data["perf_kidney"] = mapped_data["performance_score"] * mapped_data.get("kidney_disease", 0)
    comorbidities = ["diabetes", "hypertension", "heart_disease", "copd", "kidney_disease", "autoimmune", "other_comorbidity"]
    mapped_data["total_comorbidity"] = sum([mapped_data.get(c, 0) for c in comorbidities])

    # 构造 DataFrame
    df = pd.DataFrame([mapped_data])

    try:
        df["cox_risk_scaled"] = np.log1p(cox_model.predict_partial_hazard(df))
        X = pd.get_dummies(df[SURVIVAL_FEATURES], drop_first=True)
        X["cox_risk_scaled"] = df["cox_risk_scaled"]
        X = X.reindex(columns=xgb_model.get_booster().feature_names, fill_value=0)
        pred_log = xgb_model.predict(X)
        pred_months = float(np.expm1(pred_log[0]))
    except Exception as e:
        return jsonify({"error": "模型预测失败", "detail": str(e)}), 500

    return jsonify({"estimated_survival_months": round(pred_months, 2)})
