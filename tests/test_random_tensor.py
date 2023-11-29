from ..linette.random_tensor import GRVector
import numpy as np


def test_grvector_mean_tall() -> None:
    dim = 10
    mean = np.arange(dim)
    cov = np.eye(dim)
    grvector = GRVector(mean=mean, cov=cov)
    assert grvector.mean.shape[0] == dim
