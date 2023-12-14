from ast import Raise
from typing import Optional, Any, Dict
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass
from scipy.stats import multivariate_normal
from tncontract import Tensor

model_config = ConfigDict(arbitrary_types_allowed=True)
# config = ConfigDict(arbitrary_types_allowed=True)


@dataclass(config=model_config, slots=True)
class GRVector:
    """Class for handling GR vectors. All random variables will be handled as vectors under the hood."""

    mean: np.ndarray
    cov: np.ndarray
    # _rv: Optional[Any]

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

        mdim = np.squeeze(self.mean).shape[0]
        cshape = self.cov.shape

        if mdim != cshape[0]:
            raise ValueError(
                f"Mean is {mdim}-dimensional and covariance is an {cshape[0]}x{cshape[1]}-matrix. These dimensions must match."
            )
        return self


@dataclass(config=model_config, slots=True)
class GRTensor:
    mean: Tensor
    cov: Tensor
    grv: Optional[GRVector] = Field(default=None, init_var=False)

    def __post_init__(self):
        mean_vec = self.mean.copy()
        labels = mean_vec.labels
        cov_vec = self.cov.copy()

        # vectorise tensor statistics
        mean_vec.fuse_indices(labels, "i")
        mean_vec_np = mean_vec.data
        cov_vec.fuse_indices(labels, "i")
        cov_vec.fuse_indices([l + "_" for l in labels], "_i")

        # covariance will be self-adjoint so no danger of transposition errors here
        cov_vec_np = cov_vec.data

        self.grv = GRVector(mean=mean_vec_np, cov=cov_vec_np)

    @model_validator(mode="before")
    def check_tensor_labels(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure that the labels from the mean and covariance match
        i.e.
        mean.labels = ["i0","i1",...,"iN"]
        cov.labels = ["i0","i1",...,"iN","i0_","i1_",...,"iN_"]

        where the input labels math to their corresponding output with a "_" suffix
        """
        labels = values.kwargs["mean"].labels
        expected_labels = sorted(labels + [l + "_" for l in labels])
        if sorted(values.kwargs["cov"].labels) != sorted(expected_labels):
            raise ValueError(
                f"Covariance labels must match the mean labels. if mean.labels=['i0','i1',...,'iN'] then cov.labels=['i0','i1',...,'iN','i0_','i1_',...,'iN_'] is expected."
            )
        return values


def main() -> None:
    """Main function"""


if __name__ == "__main__":
    main()
