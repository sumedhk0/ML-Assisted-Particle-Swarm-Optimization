"""LightGBM wrapper for the stuck-particle classifier.

GPU training is requested via `device='cuda'` (LightGBM 4.3+). Falls back to
CPU transparently if the GPU build isn't available at runtime.
"""
import lightgbm as lgb
import numpy as np


class StuckClassifier:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.booster: lgb.Booster | None = None

    def fit(self, X_train, y_train, X_val, y_val, num_boost_round: int = 500):
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        if self.device == "cuda":
            params["device"] = "cuda"

        try:
            self.booster = lgb.train(
                params, train_set, num_boost_round=num_boost_round,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
            )
        except lgb.basic.LightGBMError as e:
            if self.device == "cuda":
                print(f"LightGBM GPU unavailable ({e}), falling back to CPU")
                params.pop("device", None)
                self.booster = lgb.train(
                    params, train_set, num_boost_round=num_boost_round,
                    valid_sets=[val_set],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
                )
            else:
                raise

    def predict_proba(self, X) -> np.ndarray:
        return self.booster.predict(X)

    def save(self, path: str):
        self.booster.save_model(path)

    def load(self, path: str):
        self.booster = lgb.Booster(model_file=path)
