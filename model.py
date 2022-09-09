import cmath
from jax import numpy as np
from jax import lax

pi = cmath.pi


class Model():
    def __init__(self, n_components=2, tol=0.1):
        self.n_components = n_components
        self.tol = tol

        self.means = None
        self.startprob = None
        self.covmat = None
        self.transmat = None

    def fit(self, data):
        data = np.ndarray(data)

