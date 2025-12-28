import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, clone
import joblib
from sklearn.feature_selection import VarianceThreshold
import warnings
import os

warnings.filterwarnings('ignore')
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from app.stacking_model import StackingClassifier



class MedicalRiskPredictor:
    def __init__(self, use_stacking=True):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.risk_mapping = {0: '低风险', 1: '中风险', 2: '高风险'}
        self.best_params = None
        self.use_stacking = use_stacking
        self.class_weights = None
        self.data_stats = None

    def load(self, data_path):
        """加载数据并使用扩展的特征集，包含更多症状特征"""
        data = pd.read_csv(data_path)
        data = data.drop(['index', 'Patient Id'], axis=1, errors='ignore')

        # 扩展特征集：风险因素 + 症状特征
        self.feature_names = [
            # 风险因素特征
            'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
            'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
            'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
            # 症状特征 - 这些对高风险更敏感
            'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
            'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
            'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
        ]

        X = data[self.feature_names]
        y = data['Level']  # 转换为0,1,2

        # 计算类别权重，关注高风险
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))

        # 计算数据统计信息
        self._compute_data_stats(data, y)

        print(f"数据形状: {data.shape}")
        print(f"使用的特征数量: {len(self.feature_names)}")
        print(f"类别分布:\n{y.value_counts().sort_index()}")
        print(f"类别权重: {self.class_weights}")

        return X, y

    def _compute_data_stats(self, data, y):
        """计算数据统计信息"""
        self.data_stats = {}
        for level in [0, 1, 2]:
            level_data = data[y == level]
            stats = {}
            for feature in self.feature_names:
                stats[feature] = {
                    'mean': level_data[feature].mean(),
                    'std': level_data[feature].std(),
                    'min': level_data[feature].min(),
                    'max': level_data[feature].max()
                }
            self.data_stats[level] = stats

    def data_quality_check(self, X, y):
        """检查数据质量问题"""
        print("\n" + "=" * 50)
        print("数据质量检查")
        print("=" * 50)

        print(f"样本数量: {len(X)}")
        print(f"特征数量: {X.shape[1]}")
        print(f"目标变量分布:\n{y.value_counts().sort_index()}")

        # 检查特征方差
        try:
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X)
            low_variance_count = sum(selector.variances_ < 0.01)
            print(f"低方差特征数量: {low_variance_count}")
        except Exception as e:
            print(f"方差检查失败: {e}")

        # 检查基准模型性能
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
            print(f"逻辑回归基准准确率: {scores.mean():.4f} (±{scores.std():.4f})")
        except Exception as e:
            print(f"基准模型检查失败: {e}")

    def explore_features(self, X_train, y_train):
        """在训练集上探索特征（避免数据泄露）"""
        print("\n" + "=" * 50)
        print("特征分析 (基于训练集)")
        print("=" * 50)

        # 转换为DataFrame以便分析
        if not isinstance(X_train, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        else:
            X_train_df = X_train

        correlation_with_target = []
        for i, col in enumerate(self.feature_names):
            try:
                # 确保y_train是numpy数组
                y_train_array = np.array(y_train)
                corr = np.corrcoef(X_train_df.iloc[:, i], y_train_array)[0, 1]
                correlation_with_target.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
            except Exception as e:
                print(f"计算特征 {col} 相关性时出错: {e}")
                correlation_with_target.append({
                    'feature': col,
                    'correlation': np.nan,
                    'abs_correlation': np.nan
                })

        correlation_df = pd.DataFrame(correlation_with_target)
        correlation_df = correlation_df.sort_values('abs_correlation', ascending=False)

        print("特征与目标的相关性 (前15):")
        print(correlation_df.head(15))

        # 特别关注症状特征
        symptom_features = [f for f in self.feature_names if f not in [
            'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
            'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
            'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker'
        ]]

        print(f"\n症状特征相关性:")
        symptom_corr = correlation_df[correlation_df['feature'].isin(symptom_features)]
        print(symptom_corr)

        return correlation_df

    def explore_data_distribution(self):
        """探索数据分布"""
        print("\n" + "=" * 50)
        print("数据分布探索")
        print("=" * 50)

        if self.data_stats is None:
            print("数据统计信息不可用")
            return

        print("高风险样本的关键特征统计:")
        key_features = ['Obesity', 'Coughing of Blood', 'Alcohol use', 'Genetic Risk',
                        'Chest Pain', 'Shortness of Breath']

        for feature in key_features:
            if feature in self.feature_names:
                low_risk_mean = self.data_stats[0][feature]['mean']
                med_risk_mean = self.data_stats[1][feature]['mean']
                high_risk_mean = self.data_stats[2][feature]['mean']
                print(
                    f"  {feature}: 低风险={low_risk_mean:.2f}, 中风险={med_risk_mean:.2f}, 高风险={high_risk_mean:.2f}")

    def check_overfitting(self, X_train, X_test, y_train, y_test):
        """检查过拟合"""
        print(f"\n过拟合检查:")
        print(f"训练集样本数: {len(X_train)}")
        print(f"测试集样本数: {len(X_test)}")

        # 简单的过拟合指标
        if len(X_train) < 100:
            print("⚠️ 训练数据较少，可能存在过拟合风险")

    def processed_data(self, data_path):
        """数据处理流程"""
        X, y = self.load(data_path)

        # 探索数据分布
        self.explore_data_distribution()

        # 数据质量检查
        self.data_quality_check(X, y)

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 分割数据 - 确保返回numpy数组
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 转换为numpy数组以避免索引问题
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # 检查过拟合
        self.check_overfitting(X_train, X_test, y_train, y_test)

        # 只在训练集上探索特征
        self.explore_features(pd.DataFrame(X_train, columns=self.feature_names), y_train)

        return X_train, X_test, y_train, y_test

    def train_stacking_model(self, data_path, n_folds=5):
        """训练Stacking集成模型 - 优化以提高高风险敏感性"""
        print("=" * 60)
        print("开始训练Stacking集成模型 (优化高风险敏感性)")
        print("=" * 60)

        X_train, X_test, y_train, y_test = self.processed_data(data_path)

        # 定义优化后的基础模型
        print("\n1. 训练优化基础模型...")

        # XGBoost参数 - 优化以关注高风险
        xgb_params = {
            'n_estimators': 300,  # 增加树的数量
            'max_depth': 8,  # 增加深度以捕捉复杂模式
            'learning_rate': 0.05,  # 降低学习率
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'scale_pos_weight': self.class_weights.get(2, 1)  # 重点关注高风险
        }

        # 随机森林参数 - 优化以关注高风险
        rf_params = {
            'n_estimators': 300,
            'max_depth': 25,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced'  # 使用平衡的类别权重
        }

        # 创建基础模型
        base_models = [
            XGBClassifier(**xgb_params),
            RandomForestClassifier(**rf_params)
        ]

        # 元模型（逻辑回归）
        meta_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            C=1.0,
            class_weight='balanced'  # 元模型也使用平衡权重
        )

        # 创建Stacking分类器
        stacking_model = StackingClassifier(
            base_models=base_models,
            meta_model=meta_model,
            use_probas=True,
            n_folds=n_folds
        )

        # 训练Stacking模型
        print("训练Stacking集成模型...")
        stacking_model.fit(X_train, y_train)
        self.model = stacking_model

        # 评估基础模型性能
        print("\n2. 评估基础模型性能...")
        base_model_scores = {}
        base_model_names = ['XGBoost', 'Random Forest']

        for i, model in enumerate(base_models):
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            base_model_scores[base_model_names[i]] = cv_scores
            print(f"{base_model_names[i]} 平均准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 评估Stacking模型
        print("\n3. 评估Stacking模型...")

        # 训练集性能
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_proba = self.model.predict_proba(X_train)
        train_loss = log_loss(y_train, train_proba)

        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"训练集损失: {train_loss:.4f}")

        # 测试集性能
        test_predictions = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_proba = self.model.predict_proba(X_test)
        test_loss = log_loss(y_test, test_proba)

        print(f"\n测试集准确率: {test_accuracy:.4f}")
        print(f"测试集损失: {test_loss:.4f}")

        # 详细分类报告
        print(f"\n{'=' * 50}")
        print("详细分类报告:")
        print(f"{'=' * 50}")
        print(classification_report(y_test, test_predictions,
                                    target_names=['低风险', '中风险', '高风险']))

        # 混淆矩阵
        cm = confusion_matrix(y_test, test_predictions)
        print(f"\n混淆矩阵:")
        print(cm)

        # 交叉验证评估
        print(f"\n4. Stacking模型交叉验证...")
        stacking_cv_scores = cross_val_score(
            stacking_model, X_train, y_train,
            cv=5, scoring='accuracy'
        )
        print(f"Stacking模型平均准确率: {stacking_cv_scores.mean():.4f} (±{stacking_cv_scores.std():.4f})")

        # 模型比较
        print(f"\n5. 模型性能比较:")
        print(f"{'模型':<15} {'平均准确率':<12} {'标准差':<10}")
        print("-" * 40)
        for name, scores in base_model_scores.items():
            print(f"{name:<15} {scores.mean():<12.4f} {scores.std():<10.4f}")
        print(f"{'STACKING':<15} {stacking_cv_scores.mean():<12.4f} {stacking_cv_scores.std():<10.4f}")

        # 高风险敏感性分析
        self._analyze_high_risk_sensitivity(X_test, y_test, test_predictions, test_proba)

        return {
            'base_model_scores': base_model_scores,
            'stacking_cv_scores': stacking_cv_scores,
            'train_accuracy': train_accuracy,
            'train_loss': train_loss,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'model': self.model
        }

    def _analyze_high_risk_sensitivity(self, X_test, y_test, test_predictions, test_proba):
        """分析高风险敏感性"""
        print(f"\n{'=' * 50}")
        print("高风险敏感性分析")
        print(f"{'=' * 50}")

        # 高风险样本的识别情况
        high_risk_indices = np.where(y_test == 2)[0]
        if len(high_risk_indices) > 0:
            high_risk_predictions = test_predictions[high_risk_indices]
            high_risk_proba = test_proba[high_risk_indices]

            high_risk_accuracy = accuracy_score(y_test[high_risk_indices], high_risk_predictions)
            print(f"高风险样本准确率: {high_risk_accuracy:.4f}")
            print(f"高风险样本数量: {len(high_risk_indices)}")

            # 高风险概率分析
            avg_high_risk_prob = np.mean(high_risk_proba[:, 2])
            print(f"高风险样本的平均高风险概率: {avg_high_risk_prob:.4f}")

            # 高风险样本的预测分布
            high_risk_pred_dist = pd.Series(high_risk_predictions).value_counts().sort_index()
            print(f"高风险样本预测分布: {dict(high_risk_pred_dist)}")
        else:
            print("测试集中没有高风险样本")

    def train_single_model(self, data_path, model_type='xgb', n_folds=5):
        """训练单个模型（XGBoost或随机森林）"""
        X_train, X_test, y_train, y_test = self.processed_data(data_path)

        if model_type == 'xgb':
            model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                random_state=42,
                scale_pos_weight=self.class_weights.get(2, 1)
            )
        elif model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        print(f"\n训练 {model_type.upper()} 模型...")

        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds, scoring='accuracy')
        print(f"{model_type.upper()} 交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 训练最终模型
        model.fit(X_train, y_train)
        self.model = model

        # 测试集评估
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_proba = model.predict_proba(X_test)
        test_loss = log_loss(y_test, test_proba)

        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"测试集损失: {test_loss:.4f}")

        # 特征重要性（如果可用）
        if hasattr(model, 'feature_importances_'):
            print(f"\n{model_type.upper()} 特征重要性 (前10):")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance.head(10))

        return {
            'cv_scores': cv_scores,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'model': model
        }

    def train_model(self, data_path, model_type='stacking', n_folds=5):
        """统一的训练接口"""
        if model_type == 'stacking':
            return self.train_stacking_model(data_path, n_folds)
        elif model_type in ['xgb', 'rf']:
            return self.train_single_model(data_path, model_type, n_folds)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def predict_risk(self, features):
        """预测新样本的风险"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train_model 方法")

        # 确保特征顺序正确
        if len(features) != len(self.feature_names):
            raise ValueError(f"特征数量不匹配，期望 {len(self.feature_names)}，得到 {len(features)}")

        # 标准化特征
        features_scaled = self.scaler.transform([features])

        # 预测
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        return {
            'risk_level': self.risk_mapping[prediction],
            'probabilities': {
                '低风险': probability[0],
                '中风险': probability[1],
                '高风险': probability[2]
            }
        }

    def create_realistic_test_cases(self):
        """基于实际数据创建现实的测试用例"""
        if self.data_stats is None:
            print("数据统计信息不可用，使用默认测试用例")
            return self._get_default_test_cases()

        # 基于实际数据统计创建测试用例
        test_cases = []

        # 高风险用例 - 基于高风险样本的统计
        high_risk_stats = self.data_stats[2]
        high_risk_features = []
        for feature in self.feature_names:
            mean_val = high_risk_stats[feature]['mean']
            # 对关键症状特征稍微提高值
            if feature in ['Coughing of Blood', 'Chest Pain', 'Shortness of Breath']:
                adjusted_val = min(mean_val + 0.5, 8)  # 稍微提高但不超过8
            else:
                adjusted_val = mean_val
            high_risk_features.append(round(adjusted_val))

        test_cases.append({
            'name': '基于数据的高风险',
            'features': high_risk_features
        })

        # 中风险用例
        med_risk_stats = self.data_stats[1]
        med_risk_features = [round(med_risk_stats[f]['mean']) for f in self.feature_names]
        test_cases.append({
            'name': '基于数据的中风险',
            'features': med_risk_features
        })

        # 低风险用例
        low_risk_stats = self.data_stats[0]
        low_risk_features = [round(low_risk_stats[f]['mean']) for f in self.feature_names]
        test_cases.append({
            'name': '基于数据的低风险',
            'features': low_risk_features
        })

        return test_cases

    def _get_default_test_cases(self):
        """获取默认测试用例"""
        return [
            {
                'name': '典型高风险-多症状',
                'features': [65, 1, 6, 6, 6, 6, 6, 6, 2, 7, 6, 6, 7, 7, 6, 6, 7, 6, 6, 6, 6, 6, 6]
            },
            {
                'name': '高风险-咳血重点',
                'features': [58, 1, 5, 5, 5, 5, 5, 5, 3, 6, 5, 5, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5]
            },
            {
                'name': '中风险-平衡',
                'features': [45, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
            },
            {
                'name': '低风险-健康',
                'features': [25, 1, 2, 2, 2, 2, 2, 2, 7, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            }
        ]

    def test_high_risk_scenarios(self):
        """测试高风险场景"""
        print(f"\n{'=' * 60}")
        print("高风险场景测试")
        print(f"{'=' * 60}")

        # 获取测试用例
        test_cases = self.create_realistic_test_cases()

        for case in test_cases:
            try:
                result = self.predict_risk(case['features'])
                high_prob = result['probabilities']['高风险']

                print(f"{case['name']}: {result['risk_level']}")
                print(
                    f"  概率分布: 低{result['probabilities']['低风险']:.3f} | 中{result['probabilities']['中风险']:.3f} | 高{high_prob:.3f}")

                # 风险评估
                if '高风险' in case['name']:
                    if result['risk_level'] == '高风险':
                        print(f"  ✅ 正确识别高风险")
                    elif high_prob > 0.4:
                        print(f"  ⚠️ 接近识别 (高风险概率: {high_prob:.3f})")
                    else:
                        print(f"  ❌ 需要调整模型")
                elif '低风险' in case['name'] and result['risk_level'] == '低风险':
                    print(f"  ✅ 正确识别低风险")

            except Exception as e:
                print(f"{case['name']} 预测失败: {e}")

    def save_model(self, filepath=None):
        """保存模型到指定路径"""
        if self.model is None:
            raise ValueError("没有训练好的模型可保存")

        # 如果没有指定路径，使用默认路径到app/models文件夹
        if filepath is None:
            # 使用绝对路径
            base_dir = r'D:\项目\肺癌预测系统后端\app\model2'
            filepath = os.path.join(base_dir, 'medical_risk_model.pkl')

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'risk_mapping': self.risk_mapping,
            'best_params': self.best_params,
            'class_weights': self.class_weights,
            'data_stats': self.data_stats
        }

        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath=None):
        """从指定路径加载模型"""
        if filepath is None:
            # 使用绝对路径
            base_dir = r'D:\项目\肺癌预测系统后端\app\model2'
            filepath = os.path.join(base_dir,'medical_risk_model.pkl')

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.risk_mapping = model_data['risk_mapping']
        self.best_params = model_data.get('best_params')
        self.class_weights = model_data.get('class_weights')
        self.data_stats = model_data.get('data_stats')

        print(f"模型已从 {filepath} 加载")


# 修改测试函数
def test_models():
    data_path = r'D:\项目\肺癌预测系统后端\data\processed\lung_cancer_processed.csv'

    print(f"\n{'=' * 60}")
    print(f"训练优化版 STACKING 模型 (包含症状特征)")
    print(f"{'=' * 60}")

    model = MedicalRiskPredictor()

    try:
        # 训练模型
        results = model.train_model(data_path, model_type='stacking', n_folds=5)

        # 保存模型到app/models文件夹
        model.save_model()

        # 测试高风险场景
        model.test_high_risk_scenarios()

        # 示例预测
        print(f"\n示例预测:")
        # 使用更典型的高风险特征
        sample_features = [65, 1, 6, 6, 6, 6, 6, 6, 2, 7, 6, 6, 7, 7, 6, 6, 7, 6, 6, 6, 6, 6, 6]
        prediction = model.predict_risk(sample_features)
        print(f"高风险样本预测结果: {prediction}")

    except Exception as e:
        print(f"STACKING 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

        # 如果Stacking失败，尝试单个模型
        print(f"\n尝试训练单个模型...")
        for model_type in ['xgb', 'rf']:
            try:
                print(f"\n训练 {model_type.upper()} 模型...")
                model = MedicalRiskPredictor()
                results = model.train_model(data_path, model_type=model_type, n_folds=5)

                # 保存模型到app/models文件夹，使用不同的文件名
                base_dir = r'D:\项目\肺癌预测系统后端\app\model2'
                filepath = os.path.join(base_dir,f'medical_risk_model_{model_type}.pkl')
                model.save_model(filepath)

                # 测试高风险场景
                model.test_high_risk_scenarios()

            except Exception as e2:
                print(f"{model_type.upper()} 训练过程中出现错误: {e2}")


if __name__ == '__main__':
    test_models()
