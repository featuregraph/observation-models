import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import json
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from lib.common.plotting.timeseries import plot_horizontal as plot

from observation_models.models.gmm import Model
from observation_models.belief.prior import Prior
from observation_models.data.load_data import load_observations

train = load_observations(problem='rod', dataset='train')
test = load_observations(problem='rod', dataset='test')
X_test = np.diff(test[:,1], prepend=0).reshape(-1,1)

model = Model(train)
pipe_sizes, probabilities = model.predict(X_test[:10])

prior = Prior(pipe_sizes, alpha=0.2)
for row in probabilities:
    prior.update(row)
    print(prior.predict(), prior.confidence().round(3))