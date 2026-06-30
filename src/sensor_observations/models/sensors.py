from sklearn.mixture import GaussianMixture
import numpy as np

def get_prediction(fit_data, predict_data):
    component_sizes = np.unique(fit_data[:, 0])
    observations = fit_data[:, 1].reshape(-1, 1)

    model = GaussianMixture(
        n_components=len(component_sizes),
        random_state=0
    ).fit(observations)

    component_order = np.argsort(model.means_.ravel())
    probabilities = model.predict_proba(predict_data).round(3)
    probabilities = probabilities[:, component_order]

    return component_sizes, probabilities
