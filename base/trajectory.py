from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from functools import cached_property
from base.boundingbox import BoundingBox, Range


@dataclass
class Trajectory:
    points: npt.NDArray[np.float32]
    times: npt.NDArray[np.float32]
    box: BoundingBox
    atom_size: float = 0.19
    traps: Optional[npt.NDArray[np.bool_]] = None
    # non_periodic_points: Optional[npt.NDArray[np.float64]] = None

    def dists(self) -> npt.NDArray[np.float32]:
        return Trajectory.extractDists(self.points_without_periodic)

    def is_intersect_borders(self) -> np.bool_:
        ppoints = self.points_without_periodic()
        xmask = np.logical_or(
            ppoints[:, 0] > self.box.xb_.max_, ppoints[:, 0] < 0
        )
        ymask = np.logical_or(
            ppoints[:, 1] > self.box.yb_.max_, ppoints[:, 1] < 0
        )
        zmask = np.logical_or(
            ppoints[:, 2] > self.box.zb_.max_, ppoints[:, 2] < 0
        )
        return np.any(xmask) or np.any(ymask) or np.any(zmask)

    def trjbox(self) -> BoundingBox:
        ppoints = self.points_without_periodic()
        mmin = ppoints.min(axis=0)
        mmax = ppoints.max(axis=0)
        tmp = [Range(i, j) for i, j in zip(mmin, mmax)]
        return BoundingBox(*tmp)

    @staticmethod
    def extractDists(
        points: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        diff = points[1:,] - points[:-1,]
        sq_diff = diff * diff
        sq_dist = np.sum(sq_diff, axis=1)
        dist = np.sqrt(sq_dist)
        return np.array(dist, dtype=np.float32)

    @cached_property
    def points_without_periodic(self) -> npt.NDArray[np.float64]:
        borders = self.box.max()
        s_2 = borders.min() / 2
        diff = self.points[1:] - self.points[:-1]
        npoints = np.zeros(shape=self.points.shape, dtype=np.float32)

        npoints[0, :] = self.points[0, :]

        shift = np.zeros(shape=(3,), dtype=np.float32)
        for i in range(diff.shape[0]):
            cdiff = diff[i]
            s_mask = np.abs(cdiff) < s_2
            ns_mask = ~s_mask
            # был справа появился слева
            neg_mask = np.logical_and(cdiff < 0, ns_mask)
            pos_mask = np.logical_and(cdiff >= 0, ns_mask)  # наоборот
            shift[s_mask] = cdiff[s_mask]
            shift[neg_mask] = cdiff[neg_mask] + borders[neg_mask]
            shift[pos_mask] = cdiff[pos_mask] - borders[pos_mask]

            npoints[i + 1, :] = npoints[i, :] + shift

        return npoints

    def count_points(self) -> int:
        return self.points.shape[0]

    @staticmethod
    def read_trajectoryes(file_name: str) -> List['Trajectory']:
        ax = []
        ay = []
        az = []

        time_steps = []
        with open(file_name) as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                time_start = line.find('t=')
                # step_start = line.find('step=')
                t = line[(time_start + 2) : (time_start + 12)]
                # step = line[step_start:]
                count = int(f.readline())

                time_steps.append(float(t))

                for i in range(count):
                    line = f.readline()
                    x, y, z = line[20:28], line[28:36], line[36:44]
                    ax.append(float(x))
                    ay.append(float(y))
                    az.append(float(z))
                if len(ax) == count:
                    line = f.readline()
                    box = BoundingBox(
                        Range(0, float(line[:10])),
                        Range(0, float(line[10:20])),
                        Range(0, float(line[20:])),
                    )
                else:
                    next(f)

        count_step = int(len(ax) / count)
        # count_step = count_step // 5
        trajectories = []
        for i in range(count):
            points = np.zeros(shape=(count_step, 3), dtype=np.float32)
            points[:, 0] = [ax[i + j * count] for j in range(count_step)]
            points[:, 1] = [ay[i + j * count] for j in range(count_step)]
            points[:, 2] = [az[i + j * count] for j in range(count_step)]

            # points = points[:(count_step // 3), :]

            trajectories.append(Trajectory(points, np.array(time_steps), box))

        return trajectories
