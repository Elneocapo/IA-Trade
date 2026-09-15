"""Red neuronal sencilla para comparar contra Random Forest."""

from __future__ import annotations

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ml_model import FEATURES


class NeuralNetworkModel:
    """MLP con escalado aprendido únicamente con los datos de entrenamiento."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=False,
        )

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "NeuralNetworkModel":
        values = self.scaler.fit_transform(frame[FEATURES])
        self.model.fit(values, target.astype(int))
        return self

    def predict(self, frame: pd.DataFrame):
        values = self.scaler.transform(frame[FEATURES])
        return self.model.predict(values)

    def predict_proba(self, frame: pd.DataFrame):
        values = self.scaler.transform(frame[FEATURES])
        return self.model.predict_proba(values)
