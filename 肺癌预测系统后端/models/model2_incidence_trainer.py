import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


FEATURE_COLUMNS = [
    '年龄',
    '吸烟年限',
    '日吸烟量',
    '家族病史',
    '空气污染指数',
    '慢性肺病',
    '癌症发病率'
]

TARGET_COLUMN = '是否患病'


def main():
    data_path = r"D:\项目\肺癌预测系统后端\data\processed\lung_ui_dataset.csv"
    model_dir = r"D:\项目\肺癌预测系统后端\app\model2"
    model_path = os.path.join(model_dir, "lung_incidence_model.pkl")

    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print("✅ 使用特征：", FEATURE_COLUMNS)
    print("✅ 样本数：", len(df))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("\n===== 模型评估 =====")
    print("ROC-AUC:", round(auc, 4))
    print(classification_report(y_test, model.predict(X_test_scaled)))

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURE_COLUMNS
        },
        model_path
    )

    print("\n✅ 模型已保存至：")
    print(model_path)


if __name__ == "__main__":
    main()
