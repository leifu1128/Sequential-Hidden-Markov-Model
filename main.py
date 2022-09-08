from cmath import pi
from jax import numpy as np
from jax import lax
import jax

class Model():
  def __init__(self, num_states, converge_thres = 0.1):
    self.num_states = num_states
    self.converge_thres = converge_thres

    self.means_ = None
    self.startprob_ = None
    self.covars_ = None
    self.transmat_ = None

    # Constants
    self.det_covars = None
    self.inv_covars = None
    self.pi = pi

  def initialize(data):
    return

  def calc_log_likelihood(slice, state):
    return (-0.5(slice - self.means_[state, 1:]).T * self.inv_covars * (slice - self.means_[state, 1:])) - lax.log((2 * self.pi * self.det_covars) ** 0.5)

  def em_pass(data, vert_length, num_features):
    new_transmat = np.zeros(vert_length, vert_length)
    new_covmat = np.zeros(vert_length, 1)
    data_mat = []
    state_seq = []
    log_likelihood = 0

    # Decode States
    likelihoods = []
    row = data[0]

    for state in range(self.num_states):
      likelihood = log(self.startprob_[state]) + self.calc_log_likelihood(row, state)
      likelihoods.append(likelihood)
      
    curr_state = np.argmax(likelihoods)
    state_seq.append(curr_state)
    log_likelihood += likelihood

    for i in range(1, vert_length):
      row = data[i]
      likelihoods = []
      prev_state = state_seq[i - 1]

      for state in range(self.num_states):
        likelihood = log(self.transmat_[prev_state][state]) + self.calc_log_likelihood(row, state)
        likelihoods.append(likelihood)
      
      curr_state = np.argmax(likelihoods)
      state_seq.append(curr_state)
      log_likelihood += likelihood
      new_transmat[prev_state][curr_state] += 1
    
    # Recalculate Transmat
    new_transmat = new_transmat / (vert_length - 1)

    # Recalculate Covariance and Means
    for state in range(self.num_states):
      ones 
      new_covmat[state] = data[]

    return log_likelihood

  def em(data):
    vert_length = len(data)
    prev_ll = 10000

    while(ll - prev_ll > self.converge_thres):
      ll = em_pass(data, vert_length)


    return

  def fit(self, data):
    self.initialize(data)
    pass
