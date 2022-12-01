import qutip as qt
import numpy as np
from scipy.stats import multivariate_normal
import tncontract as tn

class GaussianRV(object):
    """
    Abstract multivariate gaussian
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.rv = multivariate_normal(mean.reshape(-1), cov)
        super(GaussianRV, self).__init__()

    def __repr__(self):
        return "Gaussian Random Variable object. Dimension={}".format(self.rv.mean.reshape(-1).shape[0])

    def

