import argparse
import os
import pickle
import random
import sys
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from base.bufferedsampler import BufferedSampler
from processes.distribution_fitter import WeibullFitter

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.boundingbox import BoundingBox, Range
from base.trajectory import Trajectory


class Uniform01:
    def rvs(self, size: int) -> NPFArray:
        return np.random.random(size).astype(np.float32)


class UnitVector3:
    def rvs(self, size: int) -> npt.NDArray[np.float32]:
        v = np.random.normal(0.0, 1.0, size=(size, 3)).astype(np.float32)
        norm = np.linalg.norm(v, axis=1).astype(np.float32)
        # крайне редко norm=0; подстрахуемся
        norm = np.where(norm == 0, 1.0, norm).astype(np.float32)
        v /= norm[:, None]
        return v


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


class KerogenWalkSimulator:
    def __init__(
        self,
        bs_psd: BufferedSampler,
        bs_ps: BufferedSampler,
        bs_ptl: BufferedSampler,
        k,
        p,
    ):
        """
        :param self: Represent the instance of the class
        :param bs_psd: trap size distribution
        :param bs_ps: count steps in trap distribution
        :param bs_ptl: distribution of length steps between traps
        :param p: probability of moving to the next trap and 1 - p return to adjacent traps
        :param k: probability of stepping further (from the trap) and 1 - K staying in the current trap
        :return: Nothing
        """
        self.bs_psd = bs_psd
        self.bs_ps = bs_ps
        self.bs_ptl = bs_ptl
        self.p = p
        self.k = k
        self.bs_dir = BufferedSampler(UnitVector3(), "dir", size=100_000)
        self.bs_el = BufferedSampler(Uniform01(), "el", size=100_000)

    @staticmethod
    def gen_new_pos(cout_points, radius, pos):
        def gen():
            mr = radius * 1.2
            cp = 4 * cout_points
            points = np.random.uniform(-mr, mr, size=3 * cp).reshape(cp, 3)
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            dist = np.sqrt(x**2 + y**2 + z**2)
            indexes = dist < radius
            points = points[indexes, :]
            if points.shape[0] > 0:
                assert (points[0, :]).shape == pos.shape

            for i in range(3):
                points[:, i] = points[:, i] * (
                    np.array(
                        np.random.uniform(1.0, 3.0, size=points.shape[0]),
                        dtype=np.float32,
                    )
                    if random.random() < 0.4
                    else 1
                )

            points = points + pos

            return points

        points = gen()

        while points.shape[0] < cout_points:
            points = np.vstack((points, gen()))
        return points[:(cout_points), :]

    def run(self, count_points) -> Trajectory:
        traps = np.zeros(shape=(count_points - 1,), dtype=np.bool_)
        points = np.zeros(shape=(count_points, 3), dtype=np.float32)

        cur_pos_ind = 0
        cur_trap_ind = 0

        graph = nx.Graph()
        graph.add_node(cur_trap_ind, pos=[0, 0, 0], size=self.bs_psd.get())

        def move_next(cur_pos_ind, cur_trap_ind):
            lenght = self.bs_ptl.get()
            dir = self.bs_dir.get()

            pos = points[cur_pos_ind, :] + dir * lenght
            points[cur_pos_ind + 1, :] = pos

            graph.add_node(
                graph.number_of_nodes(), pos=pos, size=self.bs_psd.get()
            )
            graph.add_edge(cur_trap_ind, graph.number_of_nodes() - 1)
            traps[cur_pos_ind] = False
            return cur_pos_ind + 1, graph.number_of_nodes() - 1

        def move_to_adjacent(
            cur_pos_ind: int, cur_trap_ind: int
        ) -> Tuple[int, int]:
            neighbors = list(graph.neighbors(cur_trap_ind))
            node_num = random.choice(neighbors)
            node = graph.nodes[node_num]
            trap_pos = np.array(node['pos'])
            size = node['size']
            new_pos = KerogenWalkSimulator.gen_new_pos(1, size, trap_pos)
            # assert np.sqrt(np.sum((trap_pos - new_pos) ** 2)) <= size
            points[cur_pos_ind + 1, :] = new_pos
            traps[cur_pos_ind] = False
            return cur_pos_ind + 1, node_num

        def steps_inside(
            cur_pos_ind: int,
            cur_trap_ind: int,
            ps: BufferedSampler,
            max_count_steps: int,
        ):
            node = graph.nodes[cur_trap_ind]
            count_steps = int(ps.get())
            if count_steps == 0:
                return cur_pos_ind, cur_trap_ind
            count_steps = min(max_count_steps - cur_pos_ind - 1, count_steps)
            trap_pos = np.array(node['pos'])
            size = node['size']

            new_pos = KerogenWalkSimulator.gen_new_pos(
                count_steps, size, trap_pos
            )
            points[(cur_pos_ind + 1) : (cur_pos_ind + count_steps + 1)] = (
                new_pos
            )
            # assert np.all(np.sqrt(np.sum((new_pos - trap_pos) ** 2, axis=1)) <= size)
            traps[cur_pos_ind : (cur_pos_ind + count_steps)] = True
            return cur_pos_ind + count_steps, cur_trap_ind

        for _ in range(10):
            cur_pos_ind, cur_trap_ind = steps_inside(
                cur_pos_ind, cur_trap_ind, self.bs_ps, count_points
            )
            if cur_pos_ind + 1 == count_points:
                cur_pos_ind = 0
                points[0, :] = points[-1, :]
                points[1:, :] = 0.0
                traps[:] = False

            cur_pos_ind, cur_trap_ind = move_next(cur_pos_ind, cur_trap_ind)

        cur_pos_ind, cur_trap_ind = steps_inside(
            cur_pos_ind, cur_trap_ind, self.bs_ps, count_points
        )

        point_start_ind = cur_pos_ind

        tmp_points = points
        points = np.zeros(
            shape=(cur_pos_ind + count_points, 3), dtype=np.float32
        )
        points[:count_points, :] = tmp_points

        tmp_traps = traps
        traps = np.zeros(
            shape=(cur_pos_ind + count_points - 1,), dtype=np.bool_
        )
        traps[: (count_points - 1)] = tmp_traps

        links_start_ind = cur_pos_ind
        max_count_points = count_points + point_start_ind

        count_move_next = 0
        count_move_adj = 0
        count_iter_inside = 0
        while cur_pos_ind + 1 < max_count_points:
            if self.bs_el.get() < self.p:
                count_move_next += 1
                cur_pos_ind, cur_trap_ind = move_next(cur_pos_ind, cur_trap_ind)
            else:
                count_move_adj += 1
                cur_pos_ind, cur_trap_ind = move_to_adjacent(
                    cur_pos_ind, cur_trap_ind
                )

            if cur_pos_ind >= max_count_points:
                break

            if self.bs_el.get() <= 1 - self.k:
                count_iter_inside += 1
                cur_pos_ind, cur_trap_ind = steps_inside(
                    cur_pos_ind, cur_trap_ind, self.bs_ps, max_count_points
                )

        assert (
            points[-1, 0] != 0.0 or points[-1, 1] != 0.0 or points[-1, 2] != 0.0
        )
        points = points[point_start_ind:, :]
        traps = traps[links_start_ind:]
        mmin = points.min(axis=0) - 1e6
        mmax = points.max(axis=0) + 1e6
        df = mmax - mmin
        bbox = BoundingBox(
            *tuple(Range(k - f, l + f) for k, l, f in zip(mmin, mmax, df))
        )

        # print(f"Count move next: {count_move_next}")
        # print(f"Count move adjacent: {count_move_adj}")
        # print(f"Count iter inside: {count_iter_inside}")
        return Trajectory(points, np.arange(count_points), bbox, traps=traps)
