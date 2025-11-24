import os
from typing import Dict, Iterable, Tuple

import numpy as np
import joblib
import tensorflow as tf

from main import ANFISLayer, FEATURE_COLUMNS

CLASSES = ["Autocuidado", "Consulta Medica", "Emergencia"]


class InferencePipeline:
    """Carrega artefatos e executa previsoes em lote ou registro unico."""
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        model_path = os.path.join(artifacts_dir, "anfis_symptom_triage_model.keras")
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        imputer_path = os.path.join(artifacts_dir, "imputer.pkl")

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"ANFISLayer": ANFISLayer},
            compile=False,
        )
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)

    def _to_array(self, features: Dict[str, float] | Iterable[float]) -> np.ndarray:
        if isinstance(features, dict):
            values = [features.get(col, 0.0) for col in FEATURE_COLUMNS]
        else:
            values = list(features)
        return np.array([values], dtype=float)

    def predict(self, features: Dict[str, float] | Iterable[float]) -> Tuple[str, np.ndarray]:
        array = self._to_array(features)
        array_imputed = self.imputer.transform(array)
        array_scaled = self.scaler.transform(array_imputed)
        probs = self.model.predict(array_scaled, verbose=0)[0]
        idx = int(np.argmax(probs))
        return CLASSES[idx], probs
