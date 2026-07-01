from sklearn.mixture import GaussianMixture
import numpy as np


class Model:
    def __init__(self, fit_data):
        self.states = np.unique(fit_data[:, 0])
        self.observations = fit_data[:, 1].reshape(-1, 1)

        self.model = GaussianMixture(n_components=len(self.states), random_state=0).fit(
            self.observations
        )

    def predict(self, predict_data):
        state_order = np.argsort(self.model.means_.ravel())
        evidence = self.model.predict_proba(predict_data)
        evidence = evidence[:, state_order]

        return self.states, evidence
