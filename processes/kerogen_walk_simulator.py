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
import networkx as nx
import random

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
    def __init__(self, cdf, name, size=100000):
        assert cdf[-1, 1] == 1
        self.сdf = cdf
        self.name = name
        self.cur_index = 0
        self.size = size
        self.cur_arr = ProbDensFuncWrap.generate(cdf, size)
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
            self.cur_arr = ProbDensFuncWrap.generate(self.сdf, self.size)
            self.cur_index = 0
            print(
                f"Generated {self.name}: min={self.cur_arr.min()}, max={self.cur_arr.max()}"
            )
        v = self.cur_arr[self.cur_index]
        self.cur_index += 1
        return v

    def get_full(self):
        return ProbDensFuncWrap.generate(self.сdf, self.size)


class KerogenWalkSimulator:
    def __init__(self, psd, ps, ptl, k, p):
        """
        :param self: Represent the instance of the class
        :param psd: pore size distribution (cdf)
        :param ps: steps in trap distribution (cdf)
        :param ptl: the probability that a segment of length L is throat (cdf)
        :param p: probability of returning to the previous pore and 1 - p walking
        :param k: probability of stepping further (from the trap) and 1 - K staying in the current pore
        :return: Nothing
        :doc-author: Trelent
        """
        self.psd = ProbDensFuncWrap(psd, 'ppl')
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
                points[:, i] = points[:, i] + pos[i]
    
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
        graph.add_node(cur_trap_ind, pos=[0, 0, 0], size=self.psd.get())
        
        def move_next(cur_pos_ind, cur_trap_ind):
            lenght = self.ptl.get()
            dir = self.dir_gen.get()

            pos = points[cur_pos_ind, :] + dir * lenght
            points[cur_pos_ind + 1, :] = pos

            graph.add_node(graph.number_of_nodes(), pos=pos, size=self.psd.get())
            graph.add_edge(cur_trap_ind, graph.number_of_nodes() - 1) 
            traps[cur_pos_ind] = False
            return cur_pos_ind + 1, graph.number_of_nodes() - 1
        
        def move_to_adjacent(cur_pos_ind, cur_trap_ind):
            neighbors = list(graph.neighbors(cur_trap_ind))
            node_num = random.choice(neighbors)
            node = graph.nodes[node_num]
            trap_pos = np.array(node['pos'])
            size = node['size']
            print(trap_pos, node['size'])
            new_pos = KerogenWalkSimulator.gen_new_pos(1, size, trap_pos)
            assert np.sqrt(np.sum((trap_pos - new_pos)**2)) <= size
            points[cur_pos_ind + 1, :] = new_pos
            traps[cur_pos_ind] = False
            return cur_pos_ind + 1, node_num

        def steps_inside(cur_pos_ind: int, cur_trap_ind: int, ps: ProbDensFuncWrap, max_count_steps: int):
            node = graph.nodes[cur_trap_ind]
            count_steps = int(ps.get())
            if count_steps == 0:
                return cur_pos_ind, cur_trap_ind
            count_steps = min(max_count_steps - cur_pos_ind - 1, count_steps)
            trap_pos = np.array(node['pos'])
            size = node['size']

            points[(cur_pos_ind + 1): (cur_pos_ind + count_steps + 1)] = KerogenWalkSimulator.gen_new_pos(count_steps, size, trap_pos)
            traps[cur_pos_ind : (cur_pos_ind + count_steps)] = True
            return cur_pos_ind + count_steps, cur_trap_ind

        cur_pos_ind, cur_trap_ind = steps_inside(cur_pos_ind, cur_trap_ind, self.ps, count_points)
        cur_pos_ind, cur_trap_ind = move_next(cur_pos_ind, cur_trap_ind)
        cur_pos_ind, cur_trap_ind = steps_inside(cur_pos_ind, cur_trap_ind, self.ps, count_points)

        point_start_ind = cur_pos_ind
        
        tmp_points = points
        points = np.zeros(shape=(cur_pos_ind + count_points, 3), dtype=np.float32)
        points[:count_points, :] = tmp_points    

        tmp_traps = traps
        traps = np.zeros(shape=(cur_pos_ind + count_points - 1,), dtype=np.bool_)
        traps[:(count_points-1)] = tmp_traps    

        links_start_ind = cur_pos_ind
        max_count_points = count_points + point_start_ind

        count_move_next = 0
        count_move_adj = 0
        count_iter_inside = 0
        while cur_pos_ind + 1 < max_count_points:
            if self.el_gen.get() < self.p:
                print("move_next")
                count_move_next += 1
                cur_pos_ind, cur_trap_ind = move_next(cur_pos_ind, cur_trap_ind)
            else:
                print("move_adjacent")
                count_move_adj += 1
                cur_pos_ind, cur_trap_ind = move_to_adjacent(cur_pos_ind, cur_trap_ind)

            if cur_pos_ind >= max_count_points:
                break

            if self.el_gen.get() <= 1 - self.k:
                print("steps_inside")
                count_iter_inside += 1
                cur_pos_ind, cur_trap_ind = steps_inside(cur_pos_ind, cur_trap_ind, self.ps, max_count_points)
            
            
        assert (points[-1, 0] != 0. or points[-1, 1] != 0. or points[-1, 2] != 0.)
        points = points[point_start_ind:, :]
        traps = traps[links_start_ind:]
        mmin = points.min(axis=0) - 1e6
        mmax = points.max(axis=0) + 1e6
        df = mmax - mmin
        bbox = BoundingBox(
            *tuple(Range(k - f, l + f) for k, l, f in zip(mmin, mmax, df))
        )

        print("Nodes:")
        for num in graph.nodes:
            node = graph.nodes[num]
            print(f"Position: {node['pos']}, size: {node['size']}")
        

        # print(f"Last point: {points[-1, :]}")

        print(f"Count move next: {count_move_next}")
        print(f"Count move adjacent: {count_move_adj}")
        print(f"Count iter inside: {count_iter_inside}")
        return Trajectory(points, np.arange(count_points), bbox, traps=traps)
