import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold


class StackingClassifier(BaseEstimator, TransformerMixin):
    """Stacking集成分类器"""

    def __init__(self, base_models, meta_model, use_probas=True, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_probas = use_probas
        self.n_folds = n_folds
        self.base_models_ = []
        self.meta_model_ = None

    def fit(self, X, y):
        self.base_models_ = []
        self.meta_model_ = clone(self.meta_model)
        y = np.array(y)

        meta_features = self._generate_meta_features(X, y, training=True)
        self.meta_model_.fit(meta_features, y)
        return self

    def predict(self, X):
        meta_features = self._generate_meta_features(X, training=False)
        return self.meta_model_.predict(meta_features)

    def predict_proba(self, X):
        meta_features = self._generate_meta_features(X, training=False)
        return self.meta_model_.predict_proba(meta_features)

    def _generate_meta_features(self, X, y=None, training=True):
        if training:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            meta_features = np.zeros((X.shape[0], len(self.base_models) * 3))
            self.base_models_ = [[] for _ in range(len(self.base_models))]

            for i, model in enumerate(self.base_models):
                model_preds = np.zeros((X.shape[0], 3))

                for train_idx, val_idx in skf.split(X, y):
                    instance = clone(model)
                    instance.fit(X[train_idx], y[train_idx])
                    self.base_models_[i].append(instance)
                    model_preds[val_idx] = instance.predict_proba(X[val_idx])

                meta_features[:, i * 3:(i + 1) * 3] = model_preds

            return meta_features
        else:
            meta_features = np.zeros((X.shape[0], len(self.base_models) * 3))
            for i, models in enumerate(self.base_models_):
                avg_preds = np.mean([m.predict_proba(X) for m in models], axis=0)
                meta_features[:, i * 3:(i + 1) * 3] = avg_preds
            return meta_features
