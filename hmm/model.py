import jax
from jaxtyping import Float, Array
from jax import numpy as np
import functions


class GHMR():
    def __init__(self, n_components=3, random_seed=1, tol=0.1, max_iters=500):
        # Hyper Parameters
        self.n_components = n_components
        self.random_seed = random_seed
        self.tol = tol
        self.max_iters = max_iters

        # Model Parameters
        self.means = None
        self.covmat = None
        self.transmat = np.full((n_components, n_components), (1 / n_components))

        # Local Jitted Functions
        self.e_pass = None

    def initialize(self, data: Float[Array, "v_len h_len"]):
        # Compile Functions
        self.e_pass = jax.jit(functions.e_pass)

        # Initialise Values
        v_len, h_len = data.shape
        points = jax.random.randint(jax.random.PRNGKey(1 * self.n_components), (1, self.n_components), 0, v_len - 1)[0]
        self.means = data[points]
        self.covmat = np.zeros((self.n_components, h_len, h_len)) + np.cov(data.T)

        return

    def em_algorithim(self, data: Float[Array, "v_len h_len"]):
        ll_diff = self.tol + 1
        prev_ll = 0

        while ll_diff >= self.tol:
            self.transmat, split_data, new_ll = self.e_pass(data, self.means, self.covmat, self.transmat)
            self.means, self.covmat = functions.m_pass(split_data)
            ll_diff = new_ll - prev_ll
            prev_ll = new_ll

        return

    def fit(self, data: Float[Array, "v_len h_len"]):
        data = np.ndarray(data)
        self.initialize(data)
        self.em_algorithim(self, data)

        return
