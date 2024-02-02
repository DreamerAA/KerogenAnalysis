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

from base.boundingbox import Range, BoundingBox
from base.trajectory import Trajectory


class OneGenerator:
    def __init__(self):
        self.ind = 0
        self.s = 100000
        self.directions = self.gen()

    def gen(self):
        return np.random.uniform(0, 1, size=self.s)

    def get(self):
        if self.ind >= self.s:
            self.directions = self.gen()
            self.ind = 0
        self.ind += 1
        return self.directions[self.ind - 1]


class DirGenerator:
    def __init__(self):
        self.ind = 0
        self.s = 100000
        self.directions = self.gen()

    def gen(self):
        dir = np.random.uniform(-1, 1, size=3 * self.s).reshape(self.s, 3)
        x = dir[:, 0]
        y = dir[:, 1]
        z = dir[:, 2]
        lengthes = np.sqrt(x**2 + y**2 + z**2)
        for i in range(3):
            dir[:, i] /= lengthes
        return dir

    def get(self):
        if self.ind >= self.s:
            self.directions = self.gen()
            self.ind = 0
        self.ind += 1
        return self.directions[self.ind - 1, :]


class ProbDensFuncWrap:
    def __init__(self, pdf, name, size=100000):
        assert pdf[0, 1] == 0
        assert pdf[-1, 1] == 1
        self.pdf = pdf
        self.name = name
        self.cur_index = 0
        self.size = size
        self.cur_arr = ProbDensFuncWrap.generate(pdf, size)
        print(
            f"Generated {self.name}: min={self.cur_arr.min()}, max={self.cur_arr.max()}"
        )

    @staticmethod
    def generate(pdf, size):
        a = np.random.uniform(0, 1, size=size)
        p = pdf[:, 1]
        indexes = np.array([np.abs(p - v).argmin() for v in a])
        return pdf[indexes, 0]

    def get(self):
        if self.cur_index >= self.size:
            self.cur_arr = ProbDensFuncWrap.generate(self.pdf, self.size)
            self.cur_index = 0
            print(
                f"Generated {self.name}: min={self.cur_arr.min()}, max={self.cur_arr.max()}"
            )
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
        :param p: probability of returning to the previous pore and 1 - p walking
        :param k: probability of stepping further (from the trap) and 1 - K staying in the current pore
        :return: Nothing
        :doc-author: Trelent
        """
        self.ppl = ProbDensFuncWrap(ppl, 'ppl')
        self.ps = ProbDensFuncWrap(ps, 'ps')
        self.ptl = ProbDensFuncWrap(ptl, 'ptl')
        self.p = p
        self.k = k
        self.dir_gen = DirGenerator()
        self.el_gen = OneGenerator()

    @staticmethod
    def gen_new_pos(cout_points, radius, pos):
        def gen():
            mr = radius * 1.2
            cp = 4 * cout_points
            data = np.random.uniform(-mr, mr, size=3 * cp).reshape(cp, 3)
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            points = np.zeros(shape=(cp, 3))
            for i, a in enumerate([x, y, z]):
                points[:, i] = a + pos[i]
            dist = np.sqrt(x**2 + y**2 + z**2)
            indexes = dist < radius
            return points[indexes, :]

        points = gen()

        while points.shape[0] < cout_points:
            points = np.vstack((points, gen()))
        return points[:(cout_points), :]

    def run(self, count_points) -> Trajectory:
        points = (-1) * np.ones(shape=(count_points, 3), dtype=np.float32)
        points[0, :] = [0, 0, 0]
        d = self.ppl.get()
        pores = [(points[0, :], 0, d)]
        traps = np.zeros(shape=(count_points,), dtype=np.bool_)

        cur_ind = 0  # index of current position
        will_steps = True

        def steps_inside(cur_ind):
            cur_pore_pos, _, d = pores[-1]
            s = int(self.ps.get())

            new_pos = KerogenWalkSimulator.gen_new_pos(s, d, cur_pore_pos)
            cur_ind += 1

            size = min(count_points - cur_ind, s)
            points[cur_ind : (cur_ind + size)] = new_pos[:size]
            traps[cur_ind : (cur_ind + size)] = True
            # pores.append((cur_pore_pos, s, d))
            pores[-1] = (cur_pore_pos, s, d)
            cur_ind += s - 1
            return cur_ind, True

        cur_ind, will_steps = steps_inside(cur_ind)

        count_return = 0

        while cur_ind + 1 < count_points:
            if self.el_gen.get() < self.p and len(pores) >= 2:
                # return previous pore
                cp, _, d = pores[-2]
                new_pos = KerogenWalkSimulator.gen_new_pos(1, d, cp)
                points[cur_ind + 1, :] = new_pos
                cur_ind += 1
                count_return += 1
                will_steps = False

            if cur_ind >= count_points:
                break

            if not will_steps and self.el_gen.get() < 1 - self.k:
                # inside pore
                cur_ind, will_steps = steps_inside(cur_ind)

            if cur_ind + 1 >= count_points:
                break
            # got ot another pore
            lenght = self.ptl.get()
            dir = self.dir_gen.get()
            points[cur_ind + 1, :] = points[cur_ind, :] + dir * lenght
            d = self.ppl.get()
            pores.append((points[cur_ind + 1, :], 0, d))
            cur_ind += 1
            will_steps = False

        mmin = points.min(axis=0)
        mmax = points.max(axis=0)
        df = mmax - mmin
        bbox = BoundingBox(
            *tuple(Range(k - f, l + f) for k, l, f in zip(mmin, mmax, df))
        )

        return Trajectory(points, np.arange(count_points), bbox, traps=traps)
