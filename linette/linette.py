import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
import tncontract as tn
from dataclasses import InitVar
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, List, Tuple


model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass(config=model_config)
class Node:
    mean: tn.Tensor
    cov: tn.Tensor
    shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        self.shape = self.mean.shape


@dataclass(config=model_config, slots=True)
class Network:
    nodes: List[Node]


def main() -> None:
    """Main function"""

    # node1 = Node(mean=tn.random_tensor(2, 2), cov=tn.random_tensor(2, 2, 2, 2))
    # node2 = Node(mean=tn.random_tensor(2, 2), cov=tn.random_tensor(2, 2, 2, 2))
    # mynetwork = Network(nodes=[node1, node2])
    # print(mynetwork.nodes[0])


if __name__ == "__main__":
    main()
