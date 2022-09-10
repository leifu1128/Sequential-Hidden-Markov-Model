from jaxtyping import Array, Float, PyTree
from jax import numpy as np
from jax import lax
from jax import vmap
import jax
import cmath

pi = cmath.pi


def cll_mapped(data: Float[Array, "v_len h_len"], means: Float[Array, "num_states h_len"],
               covmat: Float[Array, "num_states h_len h_len"]):

    return vmap(jax.scipy.stats.multivariate_normal.logpdf, in_axes=(None, 0, 0),
                out_axes=1)(data, means, covmat)


@jax.jit
def e_pass(data: Float[Array, "v_len h_len"], means: Float[Array, "num_states"], covmat,
            transmat: Float[Array, "num_states"]):

    # Constants
    v_len, h_len = data.shape
    num_states = len(means)

    new_transmat = np.zeros((num_states, num_states))
    split_data = np.zeros((num_states, 1, h_len))
    likelihood = 0

    # Calculate Likelihoods
    like_mat = cll_mapped(data[:, 1:], means[:, 1:], covmat[:, 1:, 1:])
    initial_state = np.argmax(like_mat[0])

    # Decode States and Re-estimate Transition Matrix
    new_transmat = np.zeros((num_states, num_states), int)
    split_data = np.zeros((num_states, v_len - 1, h_len), float)
    likelihood = 0

    init_carry = (like_mat,
                  initial_state,
                  transmat,
                  new_transmat,
                  split_data,
                  likelihood,
                  data)

    def body(i, carry):
        like_mat_, prev_state, transmat_, new_transmat_, split_data_, likelihood_, data_ = carry
        slc = data_[i + 1]

        trans_prob = np.log(transmat_[prev_state]) + like_mat_[i]
        curr_state = np.argmax(trans_prob)
        new_transmat_.at[prev_state, curr_state].add(1)
        split_data_.at[curr_state, i].set(slc)
        likelihood_ += trans_prob[curr_state]

        return like_mat_, curr_state, transmat_, new_transmat_, split_data_, likelihood_, data_

    _, _, _, new_transmat, split_data, likelihood, _ = lax.fori_loop(0, v_len - 1, body, init_carry)

    print(new_transmat)
    print(split_data)
    split_data = split_data[:, :, 1:]
    new_transmat = new_transmat / (v_len - 1)

    return new_transmat, split_data, likelihood


def m_pass(split_data):
    split_data_ = split_data[split_data != 0]
    new_covmat = vmap(np.cov, in_axes=0)(split_data_)
    new_means = vmap(np.mean, in_axes=0)(split_data_)

    return new_means, new_covmat

