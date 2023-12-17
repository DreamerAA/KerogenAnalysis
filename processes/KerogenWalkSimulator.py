import argparse
from pathlib import Path
import sys
import os
from os import listdir
from os.path import isfile, join, dirname, realpath
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List
import numpy.typing as npt

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)


from base.trajectory import Trajectory


class ProbDensFuncWrap:
    def __init__(self, pdf, size = 100000):
        self.pdf = pdf
        self.cur_index = 0
        self.size = size
        self.cur_arr = ProbDensFuncWrap.generate(pdf, size)
        
    @staticmethod
    def generate(pdf, size):
        a = np.random.uniform(0, 1, size=size)
        p = pdf[:,1]
        indexes = np.array([np.abs(p - v).argmin() for v in a])
        return pdf[indexes, 0]

    def get(self):
        if self.cur_index >= self.size:
            self.cur_arr = ProbDensFuncWrap.generate(self.pdf, self.size)
            self.cur_index = 0
        v = self.cur_arr[self.cur_index]
        self.cur_index += 1
        return v
    
    def get_full(self):
        return ProbDensFuncWrap.generate(self.pdf, self.size)




class KerogenWalkSimulator:
    def __init__(self, ppl, ps, ptl, k, p):
        """      
        :param self: Represent the instance of the class
        :param psd: pore size distribution
        :param ps: steps in trap distribution
        :param ptl: the probability that a segment of length L is throat
        :param p: probability of returning to the previous time and 1 - p walking
        :param k: probability of stepping further (from the trap) and 1 - K staying in the time
        :return: Nothing
        :doc-author: Trelent
        """
        self.ppl = ProbDensFuncWrap(ppl)
        self.ps = ProbDensFuncWrap(ps)
        self.ptl = ProbDensFuncWrap(ptl)
        self.p = p
        self.k = k

    @staticmethod
    def gen_shifts(cout_points, radius):
        def gen():
            mr = radius*1.2
            x = np.random.uniform(-mr, mr, size=4*cout_points)
            y = np.random.uniform(-mr, mr, size=4*cout_points)
            z = np.random.uniform(-mr, mr, size=4*cout_points)
            points = np.vstack((x,y,z))
            dist = np.sqrt(x**2 + y**2 + z**2)
            indexes = dist < radius
            return points[indexes]

        points = gen()

        while points.shape[0] < cout_points:
            points = np.hstack((points, gen()))
        return points


    def run(self, count_points)->Trajectory:
        points = np.zeros(shape=(count_points,3),dtype=np.float32)
        pores = []

        cur_ind = 0
        will_steps = True

        def steps_inside():
            cur_pore_pos = points[cur_ind,:]
            d = self.ppl.get()
            s = self.ps.get()
            shifts = KerogenWalkSimulator.gen_shifts(s,d)
            points[(cur_ind+1):(cur_ind+s)] = cur_pore_pos + shifts
            pores.append((cur_pore_pos,d,s)) 
            will_steps = True

        while cur_ind + 1 < count_points:
            if np.uniform(0,1,size=1)[0] < self.p and len(pores) < 2:
                # return previous pore
                _, s, d = pores[-1]
                shifts = KerogenWalkSimulator.gen_shifts(s,d)
                points[cur_ind + 1,:] = shifts + pores[-1][0]
                cur_ind += 1

            if not will_steps and np.uniform(0,1,size=1)[0] < 1 - self.k:
                # inside pore
                steps_inside()

            # got ot another pore
            lenght = self.ptl.get()
            dxyz = np.uniform(-1,1,size=3)
            dxyz /= np.sqrt(dxyz**2)
            new_shift = dxyz*lenght
            points[cur_ind + 1,:] = points[cur_ind,:] + new_shift
            cur_ind += 1
            will_steps = False


        


