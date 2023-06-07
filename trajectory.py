
from typing import List, Optional
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from boundingbox import Range, BoundingBox

@dataclass
class Trajectory:
    points: npt.NDArray[np.float32]
    times: npt.NDArray[np.float32]
    box: BoundingBox
    atom_size: float = 0.19
    clusters: Optional[npt.NDArray[np.int32]] = None

    def dists(self)->npt.NDArray[np.float32]:
        return Trajectory.extractDists(self.points)

    @staticmethod
    def extractDists(points:npt.NDArray[np.float32])->npt.NDArray[np.float32]:
        diff = points[1:,] - points[:-1,]
        sq_diff = diff*diff
        sq_dist = np.sum(sq_diff,axis=1)
        dist = np.sqrt(sq_dist)
        return np.array(dist,dtype=np.float32)

    def points_without_periodic(self)->npt.NDArray[np.float32]:
        borders = self.box.max()
        s_2 = borders.min()/2
        diff = self.points[1:] - self.points[:-1]
        npoints = np.zeros(shape=self.points.shape,dtype=self.points.dtype)

        npoints[0,:] = self.points[0,:]
        
        shift = np.zeros(shape=(3,),dtype=npoints.dtype)
        for i in range(diff.shape[0]):
            cdiff = diff[i]
            s_mask = np.abs(cdiff) < s_2
            ns_mask = ~s_mask
            # if np.sum(ns_mask) > 0:
            #    print('f') 
            neg_mask = np.logical_and(cdiff < 0, ns_mask) # был справа появился слева
            pos_mask = np.logical_and(cdiff >= 0, ns_mask) # наоборот
            shift[s_mask] = cdiff[s_mask]
            shift[neg_mask] = cdiff[neg_mask] + borders[neg_mask]
            shift[pos_mask] = cdiff[pos_mask] - borders[pos_mask]
            
            npoints[i+1,:] = npoints[i,:] + shift

        return npoints


    @staticmethod 
    def read_trajectoryes(file_name:str)->List['Trajectory']:
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
                step_start = line.find('step=')
                t = line[(time_start+2):(time_start+12)]
                step = line[step_start:]
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
                    box = BoundingBox(Range(0, float(line[:10])),
                                     Range(0, float(line[10:20])),
                                     Range(0, float(line[20:])))
                else:
                    next(f)
            
        count_step = int(len(ax)/count)
        trajectories = []
        for i in range(count):
            points = np.zeros(shape=(count_step,3),dtype=np.float32)
            points[:,0] = [ax[i+j*count] for j in range(count_step)]
            points[:,1] = [ay[i+j*count] for j in range(count_step)]
            points[:,2] = [az[i+j*count] for j in range(count_step)]
            
            trajectories.append(Trajectory(points, np.array(time_steps), box))

        return trajectories
        