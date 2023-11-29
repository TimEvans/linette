from token import OP
from typing import Tuple, Optional, Any
import numpy as np
from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    model_validator,
    ConfigDict,
)
from scipy.stats import multivariate_normal


# config = ConfigDict(arbitrary_types_allowed=True)


class GRVector(BaseModel):
    """Class for handling GR vectors. All random variables will be handled as vectors under the hood."""

    mean: np.ndarray
    cov: np.ndarray
    _rv: Optional[Any]

    class Config:
        """Pydantic config class"""

        arbitrary_types_allowed = True

    @field_validator("cov", mode="before")
    @classmethod
    def check_cov_square(cls, v) -> np.ndarray:
        """Ensure that the covariance matrix is square"""

        shape = v.shape
        if not shape[0] == shape[1]:
            raise ValueError("Covariance is not square")
        return v

    @field_validator("cov", mode="before")
    @classmethod
    def check_cov_symmetric(cls, v) -> np.ndarray:
        """Ensure that the covariance matrix is symmetric"""

        if not np.allclose(v, v.T, 1e-8):
            raise ValueError("Covariance is not symmetric")
        return v

    @field_validator("cov", mode="before")
    @classmethod
    def check_cov_pos(cls, v) -> np.ndarray:
        """Ensure that the covariance matrix is positive definite"""

        _, s, _ = np.linalg.svd(v)
        if not np.all(s > 0):
            raise ValueError("Covariance is not symmetric")
        return v

    @field_validator("mean", mode="before")
    @classmethod
    def check_mean_tall(cls, v) -> np.ndarray:
        """By convention, ensure that the mean is an nx1-array."""

        ndim = np.squeeze(v).ndim
        if ndim > 1:
            raise ValueError(f"Mean is {ndim}-dimensional, must be 1-dimensional")
        return v.reshape(-1, 1)

    @model_validator(mode="after")
    # @classmethod
    def check_dim_match(self) -> "GRVector":
        """Ensure that the covariance matrix and mean dimensions match"""

        mdim = self.mean.shape[0]
        cshape = self.cov.shape

        if mdim != cshape[0]:
            raise ValueError(
                f"Mean is {mdim}-dimensional and covariance is an {cshape[0]}x{cshape[1]}-matrix. These dimensions must match."
            )
        self._rv = multivariate_normal(mean=self.mean.reshape(-1), cov=self.cov)
        return self


# def main() -> None:
#     """Main function"""


# if __name__ == "__main__":
#     main()
