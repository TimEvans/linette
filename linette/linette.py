import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
import tncontract as tn
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, List


class Node(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: np.ndarray


class Network(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    nodes: List[Node]


def main() -> None:
    """Main function"""
    mynodes = [Node(data=np.random.randn(2, 2)) for _ in range(5)]
    mynetwork = Network(nodes=mynodes)
    print(mynetwork.nodes[0])


if __name__ == "__main__":
    main()
