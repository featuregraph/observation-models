import numpy as np


class Prior:
    def __init__(self, states, alpha=0.2, initial=None):
        self.states = states
        self.alpha = alpha

        if initial is None:
            self.value = np.ones(len(states)) / len(states)
        else:
            self.value = initial / initial.sum()

    def update(self, evidence, valid=True):
        if valid:
            self.value = (1 - self.alpha) * self.value + self.alpha * evidence
            self.value = self.value / self.value.sum()

        return {
            "belief": self.value,
            "prediction": self.predict(),
            "confidence": self.confidence(),
            "valid": valid,
        }

    def predict(self):
        return self.states[np.argmax(self.value)]

    def confidence(self):
        return self.value.max()
