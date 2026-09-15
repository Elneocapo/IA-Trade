"""Red neuronal sencilla con entrenamiento por épocas y curva de aprendizaje."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ml_model import FEATURES


class NeuralNetworkModel:
    """MLP entrenada por épocas, con seguimiento de pérdida y validación."""

    def __init__(self, epochs: int = 300) -> None:
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.model = self._new_model()
        self.loss_history: list[float] = []
        self.validation_history: list[float] = []

    @staticmethod
    def _new_model() -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=1,
            warm_start=True,
            shuffle=True,
            random_state=42,
        )

    def fit(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
        show_plot: bool = True,
    ) -> "NeuralNetworkModel":
        """Entrena durante un número fijo de épocas y muestra su evolución."""
        values = self.scaler.fit_transform(frame[FEATURES])
        target_values = target.astype(int).to_numpy()

        # Validación interna cronológica: nunca toca el test final.
        split = max(int(len(values) * 0.85), len(values) - 100)
        split = min(split, len(values) - 20)
        train_values = values[:split]
        train_target = target_values[:split]
        validation_values = values[split:]
        validation_target = target_values[split:]

        self.model = self._new_model()
        self.loss_history = []
        self.validation_history = []

        for epoch in range(self.epochs):
            self.model.fit(train_values, train_target)
            self.loss_history.append(float(self.model.loss_))
            validation_prediction = self.model.predict(validation_values)
            self.validation_history.append(
                float(
                    balanced_accuracy_score(
                        validation_target, validation_prediction
                    )
                )
            )

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(
                    f"  Época {epoch + 1:>3}/{self.epochs} | "
                    f"loss {self.loss_history[-1]:.4f} | "
                    f"val. balanced {self.validation_history[-1] * 100:.2f}%"
                )

        if show_plot:
            self._show_training_plot()

        # Entrenamiento final con todos los datos del conjunto de entrenamiento.
        # El test final permanece completamente separado.
        self.model = self._new_model()
        for _ in range(self.epochs):
            self.model.fit(values, target_values)

        return self

    def _show_training_plot(self) -> None:
        epochs = np.arange(1, len(self.loss_history) + 1)
        figure, axis_loss = plt.subplots(figsize=(9, 5))
        axis_loss.plot(epochs, self.loss_history, label="Loss de entrenamiento")
        axis_loss.set_xlabel("Época")
        axis_loss.set_ylabel("Loss")
        axis_loss.grid(True, alpha=0.25)

        axis_validation = axis_loss.twinx()
        axis_validation.plot(
            epochs,
            np.array(self.validation_history) * 100,
            label="Validación balanceada",
            linestyle="--",
        )
        axis_validation.set_ylabel("Balanced accuracy (%)")
        axis_validation.set_ylim(0, 100)

        figure.suptitle("Entrenamiento de la red neuronal")
        figure.tight_layout()
        plt.show()
        plt.close(figure)

    def predict(self, frame: pd.DataFrame):
        values = self.scaler.transform(frame[FEATURES])
        return self.model.predict(values)

    def predict_proba(self, frame: pd.DataFrame):
        values = self.scaler.transform(frame[FEATURES])
        return self.model.predict_proba(values)
