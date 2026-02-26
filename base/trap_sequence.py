from utils.types import NPIArray, NPFArray
from dataclasses import dataclass


@dataclass
class TrapSequence:
    traps: NPIArray
    times: NPFArray
