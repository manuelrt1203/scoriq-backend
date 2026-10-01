"""Classes partagées entre calibrate_v2.py et predict_v3.py."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

# Probabilité minimale par issue après calibration. L'isotonique renvoie parfois
# exactement 0 (issue « impossible »), ce qui fausse les cotes justes et le value.
MIN_PROBA = 0.02


def plancher_probas(probas, floor=MIN_PROBA):
    """Normalise des probas (n, k) puis les ramène dans [floor, 1] : p' = floor + (1 - k·floor)·p.
    Transformation affine : l'ordre des issues et la somme à 1 sont conservés."""
    probas = np.asarray(probas, dtype=float)
    sums = probas.sum(axis=-1, keepdims=True)
    k = probas.shape[-1]
    probas = np.where(sums > 0, probas / np.where(sums == 0, 1.0, sums), 1.0 / k)
    return floor + (1 - k * floor) * probas


class LeagueCalibrator:
    """
    Calibrateur isotonique 3-classes pour une ligue.
    Corrige les biais systématiques (ex : sous-estimation du nul).
    """
    def __init__(self):
        self.iso = [IsotonicRegression(out_of_bounds="clip") for _ in range(3)]

    def fit(self, probas, y):
        for cls in range(3):
            self.iso[cls].fit(probas[:, cls], (y == cls).astype(float))

    def predict(self, probas):
        corrected = np.stack(
            [self.iso[cls].predict(probas[:, cls]) for cls in range(3)], axis=1
        )
        return plancher_probas(corrected)

    def predict_one(self, proba_list):
        p = np.array(proba_list).reshape(1, -1)
        return self.predict(p)[0].tolist()
