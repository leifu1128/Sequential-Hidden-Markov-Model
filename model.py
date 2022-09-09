from cmath import pi
from jax import numpy as np
from jax import lax

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