import os
import joblib
import numpy as np


class LungCancerIncidenceService:
    def __init__(self):
        model_path = r"D:\项目\肺癌预测系统后端\app\model2\lung_incidence_model.pkl"

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = bundle["features"]

    def predict_incidence_probability(self, input_data: dict) -> float:
        """
        input_data 示例：
        {
            "年龄": 55,
            "吸烟年限": 30,
            "日吸烟量": 20,
            "家族病史": 1,
            "空气污染指数": 75,
            "慢性肺病": 1,
            "癌症发病率": 0.032
        }
        """
        try:
            x = [input_data[f] for f in self.features]
        except KeyError as e:
            raise ValueError(f"❌ 缺少必要特征: {e}")

        x = np.array(x).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        proba = self.model.predict_proba(x_scaled)[0][1]
        return round(float(proba), 4)
