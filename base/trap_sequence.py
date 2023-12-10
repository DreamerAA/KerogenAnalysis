from typing import List, Tuple, Any
import numpy.typing as npt
from base.trajectory import Trajectory
import numpy as np
from dataclasses import dataclass


@dataclass
class TrapSequence:
    points: npt.NDArray[np.float32]
    times: npt.NDArray[np.float32]
        
