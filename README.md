# 肺癌智能辅助预测系统

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)](https://xgboost.readthedocs.io/)

---

## 项目简介

肺癌智能辅助预测系统旨在为临床医生提供基于患者多维度特征的生存时间预测。系统结合 **Cox 回归模型** 与 **XGBoost 回归模型**，综合患者病史、生命体征和实验室数据，实现个性化的肺癌生存风险评估。

主要功能：

- 基于患者临床特征预测肺癌生存时间（月）
- 生成患者风险评分（Cox 模型）
- 可视化特征重要性
- 支持前端 Web 调用 API

---

## 功能接口示例

### 健康检查接口

```http
GET /api/health
````

### 生存时间预测接口

```http
POST /api/predict/survival
Content-Type: application/json
```

#### 请求示例

```json
{
  "race": 1,
  "insurance": 1,
  "family_history": 0,
  "diabetes": 0,
  "hypertension": 1,
  "heart_disease": 0,
  "copd": 1,
  "kidney_disease": 0,
  "autoimmune": 0,
  "performance_score": 80,
  "bp_stage": 2,
  "other_comorbidity": 0,
  "total_comorbidity": 2,
  "perf_copd": 80,
  "perf_kidney": 0
}
```

#### 返回示例

```json
{
  "predicted_survival_months": 34.56
}
```

---

## 技术栈

* **后端**：Python, Flask
* **机器学习**：Lifelines (CoxPH), XGBoost
* **前端**：React + TypeScript
* **数据处理**：Pandas, NumPy
* **可视化**：Matplotlib
* **模型保存**：Joblib

---

## 主要项目结构

```text
lung-cancer-prediction/
├─ 肺癌预测系统后端/
│  ├─ app/
│  │  ├─ __init__.py
│  │  ├─ app_loader.py
│  │  ├─ model_loader.py
│  │  ├─ routes.py
│  │  ├─ stacking_model.py
│  │  └─ model2/
│  │      ├─ cox_model.pkl
│  │      ├─ xgb_survival_model.pkl
│  │      ├─ lung_incidence_model.pkl
│  │      └─ medical_risk_model.pkl
│  ├─ data/
│  │  ├─ raw/
│  │  │  ├─ lung_cancer_processed.csv
│  │  │  ├─ lung_ui_dataset.csv
│  │  │  └─ simulated_lung_cancer_survival.csv
│  │  └─ processed/
│  │      ├─ cancer_patient_data_sets.csv
│  │      ├─ Lung_Cancer_Dataset.csv
│  │      └─ simulated_lung_cancer_survival.csv
│  └─ 可视化/
│      ├─ model1_data1.ipynb
│      └─ model2_data.ipynb
├─ run.py
├─ requirements.txt
└─ README.md

```

---

## 安装与运行

### 1. 克隆项目

```bash
git clone https://github.com/summer666623/lung-cancer-prediction.git
cd lung-cancer-prediction
```

### 2. 安装依赖（推荐使用虚拟环境）

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. 启动后端服务

```bash
python run.py
```

### 4. 前端项目启动（Vite）

```bash
cd src
npm install
npm run dev
```

---

## 模型训练说明

核心模型：

* **Cox 回归模型**：计算患者风险评分
* **XGBoost 回归模型**：在 Cox 风险基础上预测生存时间

训练脚本位于 `app/models/`，训练完成后模型存储在 `app/model2/`。

---

## 联系方式

作者：summer666623
邮箱：[2947142640@qq.com](mailto:2947142640@qq.com)
欢迎交流与合作！

