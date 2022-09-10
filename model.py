import cmath
from jax import numpy as np
from jax import lax
import functions

pi = cmath.pi


class GHMR():
    def __init__(self, n_components=2, tol=0.1):
        self.n_components = n_components
        self.tol = tol

        self.means = None
        self.covmat = None
        self.transmat = None

    def initialize(self, data):
        return init_ll

    def em_algorithim(self, data, init_ll):
        ll_diff = 10000
        prev_ll = init_ll

        while ll_diff >= self.tol:
            self.transmat, split_data, new_ll = functions.e_pass(data, self.means, self.covmat, self.transmat)
            self.means, self.covmat = functions.m_pass(split_data)
            ll_diff = new_ll - prev_ll
            prev_ll = new_ll

        return

    def fit(self, data):
        data = np.ndarray(data)
        init_ll = self.initialize(data)
        self.em_algorithim(self, data, init_ll)

        return

    def predict(self):
        pass