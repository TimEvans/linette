from token import OP
from typing import Tuple, Optional
import numpy as np
from pydantic import BaseModel, ValidationError, field_validator


class GRVector(BaseModel):
    mean: np.ndarray
    cov: np.ndarray
    # _rv: Optional[BaseModel]

    # def __init__(self, **data):
    #     data["shape"] = data["data"].shape
    #     super().__init__(**data)

    class Config:
        """Pydantic config class"""

        arbitrary_types_allowed = True

    @field_validator("cov", mode="after")
    def cov_symmetric(cls, v):
        if not np.allclose(v, v.T, 1e-8):
            raise ValueError("Covariance is not symmetric")
        return v

    @field_validator("mean", mode="after")
    def mean_tall(cls, v):
        ndim = np.squeeze(v).ndim
        if ndim > 1:
            raise ValueError(f"Mean is {ndim}-dimensional, must be 1-dimensional")
        return v.reshape(-1, 1)
