"""Classes partagées entre calibrate_v2.py et predict_v3.py."""

import numpy as np
from sklearn.isotonic import IsotonicRegression


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
        row_sums = corrected.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return corrected / row_sums

    def predict_one(self, proba_list):
        p = np.array(proba_list).reshape(1, -1)
        return self.predict(p)[0].tolist()
