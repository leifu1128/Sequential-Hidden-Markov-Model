from jaxtyping import Array, Float, PyTree
from jax import numpy as np
from jax import lax
from jax import vmap
import cmath

pi = cmath.pi


def calc_log_likelihood(slc: Float[Array], mean: float, inv_covar: float, det_covar: float):
    return (-0.5 * np.transpose(slc - mean) * inv_covar * (slc - mean)) \
           - lax.log((2 * pi * det_covar) ** 0.5)


def em_pass(data: Float[Array], means: Float[Array, "num_states"], inv_covars: Float[Array, "num_states"], det_covars:
Float[Array, "num_states"], transmat: Float[Array, "num_states"]):
    # Constants
    v_len, h_len = data.shape
    num_states = len(means)

    new_transmat = np.zeros(num_states, num_states)
    split_data = np.zeros(num_states, 1, h_len)
    likelihood = 0

    # Calculate Likelihoods
    ll_map = vmap(calc_log_likelihood(), in_axes=(None, 0, 0, 0))
    likmat = vmap(ll_map, in_axes=(0, None, None, None), out_axes=0)(data[:, 1:], means[:, 1:], inv_covars, det_covars)
    initial_state = np.argmax(likmat[0])

    # Decode States and Recalculate Parameters
    new_transmat = np.zeros(num_states, num_states)
    split_data = np.zeros(num_states, 1, h_len)
    likelihood = 0

    init_carry = (likmat,
                  initial_state,
                  transmat,
                  new_transmat,
                  split_data,
                  likelihood,
                  data)

    def body(i, carry):
        likmat_, prev_state, transmat_, new_transmat_, split_data_, likelihood_, data_ = carry
        slc = data_[i + 1]

        trans_prob = np.log(transmat_) + likmat_[i]
        curr_state = np.argmax(trans_prob)
        new_transmat_[prev_state][curr_state] += 1
        split_data_[curr_state] = np.vstack(split_data_[curr_state], slc)
        likelihood_ += trans_prob[curr_state]

        return curr_state, transmat_, new_transmat_, split_data_, likelihood_, data_

    _, _, num_states, split_data, likelihood, _ = lax.fori_loop(0, v_len - 1, body(), init_carry)

    split_data = split_data[:, :, 1:]

    new_covmat = vmap(np.cov(), in_axes=2)(split_data)


def expectation_maximization(data: Float[Array], tol, covmat):
    inv_covars = np.linalg.inv(covmat[1:, 1:])
    det_covars = np.linalg.det(covmat[1:, 1:])
