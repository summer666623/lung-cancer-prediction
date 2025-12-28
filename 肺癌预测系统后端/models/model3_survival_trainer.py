import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, plot_importance
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import joblib
import os

# =========================
# 1. 读取仿真数据
# =========================
DATA_PATH = r"D:\项目\肺癌预测系统后端\data\processed\simulated_lung_cancer_survival.csv"
df = pd.read_csv(DATA_PATH)

TARGET_TIME = "time"
TARGET_EVENT = "event"

FEATURES = [
    "race","insurance","family_history","diabetes","hypertension",
    "heart_disease","copd","kidney_disease","autoimmune",
    "performance_score","bp_stage","other_comorbidity","total_comorbidity",
    "perf_copd","perf_kidney"
]

# =========================
# 2. 训练/测试集划分
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET_EVENT]
)

# =========================
# 3. Cox 模型训练
# =========================
cox = CoxPHFitter(penalizer=0.01)
cox.fit(train_df, duration_col=TARGET_TIME, event_col=TARGET_EVENT)
print("\n===== Cox 回归结果 =====")
cox.print_summary()

# =========================
# 4. Cox 风险计算并确保方向正确
# 高风险 → 生存时间短
# =========================
train_risk = cox.predict_partial_hazard(train_df)
test_risk = cox.predict_partial_hazard(test_df)

# 如果高风险 → 生存短，应该取负数
train_df["cox_risk_scaled"] = np.log1p(-train_risk + 1e-6)
test_df["cox_risk_scaled"] = np.log1p(-test_risk + 1e-6)

# =========================
# 5. XGBoost 特征准备
# =========================
X_train = pd.get_dummies(train_df[FEATURES], drop_first=True)
X_train["cox_risk_scaled"] = train_df["cox_risk_scaled"]

X_test = pd.get_dummies(test_df[FEATURES], drop_first=True)
X_test["cox_risk_scaled"] = test_df["cox_risk_scaled"]
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 预测生存时间 log 缩放
y_train = np.log1p(train_df[TARGET_TIME])
y_test = np.log1p(test_df[TARGET_TIME])

# =========================
# 6. XGBoost 回归训练
# =========================
xgb = XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)
xgb.fit(X_train, y_train)

# =========================
# 7. 性能评估
# =========================
pred_log = xgb.predict(X_test)
pred_time = np.expm1(pred_log)
mae = mean_absolute_error(test_df[TARGET_TIME], pred_time)
print(f"\n===== XGBoost 生存时间预测 =====")
print(f"MAE（越低越好）: {mae:.2f} 月")

# =========================
# 8. 特征重要性保存
# =========================
plt.figure(figsize=(10,6))
plot_importance(xgb, max_num_features=15)
plt.title("Top 15 Feature Importance")
plt.savefig("feature_importance.png")
plt.close()
print("✅ 特征重要性图已保存：feature_importance.png")

# =========================
# 9. 示例患者预测
# =========================
example_data = pd.DataFrame([
    {
        "race": 1,"insurance": 1,"family_history": 0,"diabetes": 0,
        "hypertension": 1,"heart_disease": 0,"copd": 1,"kidney_disease": 0,
        "autoimmune": 0,"performance_score": 80,"bp_stage": 2,
        "other_comorbidity": 0,"total_comorbidity": 2,
        "perf_copd": 80*1,"perf_kidney": 80*0
    },
    {
        "race": 2,"insurance": 0,"family_history": 1,"diabetes": 1,
        "hypertension": 1,"heart_disease": 1,"copd": 0,"kidney_disease": 1,
        "autoimmune": 0,"performance_score": 60,"bp_stage": 3,
        "other_comorbidity": 1,"total_comorbidity": 4,
        "perf_copd": 60*0,"perf_kidney": 60*1
    }
])

example_risk = cox.predict_partial_hazard(example_data)
example_data["cox_risk_scaled"] = np.log1p(-example_risk + 1e-6)

example_X = pd.get_dummies(example_data[FEATURES], drop_first=True)
example_X["cox_risk_scaled"] = example_data["cox_risk_scaled"]
example_X = example_X.reindex(columns=X_train.columns, fill_value=0)

pred_example_log = xgb.predict(example_X)
pred_example_time = np.expm1(pred_example_log)

for i, t in enumerate(pred_example_time):
    print(f"患者 {i+1} 预测生存时间（月）：{t:.2f}")
    print(f"  Cox 风险分数 (log 缩放): {example_data['cox_risk_scaled'].iloc[i]:.3f}")

# =========================
# 10. 模型保存
# =========================
MODEL_DIR = r"D:\项目\肺癌预测系统后端\app\model2"
os.makedirs(MODEL_DIR, exist_ok=True)

cox_path = os.path.join(MODEL_DIR, "cox_model.pkl")
joblib.dump(cox, cox_path)
print(f"✅ Cox 模型已保存: {cox_path}")

xgb_path = os.path.join(MODEL_DIR, "xgb_survival_model.pkl")
joblib.dump(xgb, xgb_path)
print(f"✅ XGBoost 模型已保存: {xgb_path}")
