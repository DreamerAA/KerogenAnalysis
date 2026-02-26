from base.trajectory import Trajectory
from base.trap_sequence import TrapSequence
import numpy as np


class TrapExtractor:
    @staticmethod
    def get_trap_seq(edge_traps: NPBArray, delta_time: float) -> TrapSequence:
        data = [(0.0, 1)]
        cur_time = 0.0
        count_points = 0
        for i in range(1, len(edge_traps)):
            p, n = edge_traps[i - 1], edge_traps[i]
            if n:
                cur_time += delta_time
                count_points += 1
            elif p:
                data.append((cur_time, count_points + 1))
                cur_time = 0
                count_points = 0
            elif not p:
                data.append((0.0, 1))

        if cur_time > 1e-300:
            data.append((cur_time, count_points + 1))

        times = np.array([d[0] for d in data])
        traps = np.array([d[1] for d in data])

        return TrapSequence(traps, times)

    @staticmethod
    def get_zero_trap_probability(seq: TrapSequence):
        return np.count_nonzero(seq.times == 0) / len(seq.times)
