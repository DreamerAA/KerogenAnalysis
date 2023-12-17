from typing import List, Tuple, Any
import numpy as np
import numpy.typing as npt
from base.trajectory import Trajectory
from .trajectory_analyzer import TrajectoryAnalizer
from base.trap_sequence import TrapSequence


class TrapExtractor:
    def __init__(self, analyzer: TrajectoryAnalizer):
        self.analyzer = analyzer

    def run(self, trj:Trajectory, run_func) -> TrapSequence:
        traps_result = run_func(self.analyzer, trj)
        trj.traps = traps_result
        return self.get_trap_seq(trj)

    def get_trap_seq(self, trj:Trajectory):
        edge_traps = trj.traps
        dt = trj.delta_time*1e-12

        points = trj.points_without_periodic

        data = []
        cur_time = 0.
        pos = points[0,:]
        count_points = 0
        for i, trap in enumerate(edge_traps):
            if trap:
                cur_time += dt
                pos = pos + points[i,:]
                count_points += 1
            else:
                data.append((pos,cur_time))
                pos = points[i,:]/count_points
                count_points = 0
                cur_time = 0

            
        points = np.zeros((len(data),3),dtype=np.float32)
        for i,d in enumerate(data):
            points[i,:] += d[0]
        times = np.array([d[1] for d in data])
        
        return TrapSequence(points, times)