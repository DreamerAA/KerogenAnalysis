import sys
import os
from pathlib import Path
from os import listdir
from os.path import realpath, join, isfile
import time
from typing import List, Tuple
import numpy as np
import argparse
import networkx as nx
import json
from scipy import ndimage
import subprocess

from base.boundingbox import BoundingBox, Range
from base.reader import Reader
from visualizer.visualizer import Visualizer


def read_and_draw_pnm_and_img(pnm_path: str, path_to_img: str):
    r, tl, ll, positions = Reader.read_pnm_ext_data(pnm_path)

    colors_data = {
        0: (1.0, 0.0, 0.0, 1.0),
    }
    scales_data = {i: rad for i, rad in enumerate(r)}

    node_pos = {idx: pos for idx, pos in enumerate(positions)}

    bord = positions.max(axis=0)
    bbox = BoundingBox(
        Range(0, bord[0]),
        Range(0, bord[1]),
        Range(0, bord[2]),
    )

    graph = nx.Graph()
    graph.add_nodes_from(
        [(i, {"color_id": 0, "scale_id": i}) for i, _ in enumerate(r)]
    )
    graph.add_weighted_edges_from(
        [(l[0], l[1], tl[i, 0]) for i, l in enumerate(ll)]
    )

    float_img = np.load(path_to_img)
    float_img = ndimage.gaussian_filter(float_img, 4)
    float_img = np.pad(float_img, [(1, 1), (1, 1), (1, 1)], 'maximum')
    float_img[:, (float_img.shape[1] // 2) :, :] = 10.0 * float_img.max()

    rad = tl[:, 0].mean()

    Visualizer.draw_graph_and_img(  # type: ignore
        graph,
        float_img,
        node_pos,
        0.11,
        bbox,
        size_node=0.8,
        size_edge=rad / 2,
        colors_data=colors_data,
        scales_data=scales_data,
        scale='non',
    )
    Visualizer.show()


if "__main__" == __name__:
    main_path = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"

    # main_name = "num=25000_time-ps=50_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000"
    # main_name = "num=551025000_time-ps=1102050_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000"
    main_name = "num=1102025000_time-ps=2204050_bbox=(x=(1.489-4.742)_y=(2.078-5.332)_z=(4.881-8.134))_resolution=0.013015000"

    fimg_path = join(
        main_path, "float_images", "result-img-" + main_name + ".npy"
    )
    pnm_prefix = join(main_path, "pnm", "pnm-" + main_name)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fimg_path',
        type=str,
        default=fimg_path,
    )
    parser.add_argument(
        '--pnm_prefix',
        type=str,
        default=pnm_prefix,
    )

    args = parser.parse_args()

    # extanded_struct_extr(
    #     args.structure_path,
    #     args.save_path,
    #     indexes,
    #     True,
    #     200,
    # )

    # run_extractor(ker_prefix, args.extractor_path, args.config_extractor_path)

    read_and_draw_pnm_and_img(
        args.pnm_prefix,
        args.fimg_path,
    )
