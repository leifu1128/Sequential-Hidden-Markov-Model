from jaxtyping import Array, Float, PyTree
from jax import numpy as np
from jax import lax
from jax import vmap
import jax


def cll_mapped(data: Float[Array, "v_len h_len"], means: Float[Array, "num_states h_len"],
               covmat: Float[Array, "num_states h_len h_len"]):

    return vmap(jax.scipy.stats.multivariate_normal.logpdf, in_axes=(None, 0, 0),
                out_axes=1)(data, means, covmat)


def e_body(i, carry):
    like_mat_, transmat_, new_transmat_, state_sequence_, likelihood_, data_ = carry

    prev_state = state_sequence_[i]
    slc = data_[i + 1, :]
    trans_prob = np.log(transmat_[prev_state]) + like_mat_[i, :]
    curr_state = np.argmax(trans_prob)
    new_transmat_ = new_transmat_.at[prev_state, curr_state].add(1)
    state_sequence_ = state_sequence_.at[i + 1].set(curr_state)
    likelihood_ += (trans_prob[curr_state])

    return like_mat_, transmat_, new_transmat_, state_sequence_, likelihood_, data_


def e_pass(data: Float[Array, "v_len h_len"], means: Float[Array, "num_states h_len"],
           covmat: Float[Array, "num_states h_len h_len"], trans_mat: Float[Array, "num_states num_states"]):

    # Constants
    v_len, h_len = data.shape
    num_states = len(means)

    # Calculate Likelihoods
    like_mat = cll_mapped(data[:, 1:], means[:, 1:], covmat[:, 1:, 1:])
    initial_state = np.argmax(like_mat[0])

    # Decode States and Re-estimate Transition Matrix
    new_transmat = np.zeros((num_states, num_states), float)
    state_sequence = np.zeros(v_len, int)
    likelihood = float(0)

    init_carry = (like_mat, trans_mat, new_transmat, state_sequence, likelihood, data)

    _, _, new_transmat, state_sequence, likelihood, _ = lax.fori_loop(0, v_len - 1, e_body, init_carry)

    new_transmat = new_transmat.T / new_transmat.sum(axis=1)

    return new_transmat.T, state_sequence, likelihood


def m_pass(data: Float[Array, "v_len h_len"], state_sequence: Float[Array, "v_len"], n_components: int):
    for i in range(n_components):
        inds = np.argwhere(state_sequence == i).flatten()
        block = data[inds[inds != 0], :]

        cov_block = np.cov(block.T)[np.newaxis, :, :]
        mean_block = np.mean(block, axis=0)[np.newaxis, :]

        if i == 0:
            new_covmat = cov_block
            new_means = mean_block

        else:
            new_covmat = np.vstack((new_covmat, cov_block))
            new_means = np.vstack((new_means, mean_block))

    return new_means, new_covmat
