import argparse
import json
import re
import subprocess
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path

from utils.utils import kprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_extractor',
        type=str,
        default="/media/andrey/Samsung_T5/DCore/SSM-2/pore-network-extraction/build/clang-15-release-cpu/bin/extractor_example",
    )
    parser.add_argument(
        '--default_path',
        type=str,
        default="/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/",
    )
    parser.add_argument(
        '--path_to_config',
        type=str,
        default="/media/andrey/Samsung_T5/DCore/SSM-2/pore-network-extraction/example/config/ExtractorExampleConfig.json",
    )
    parser.add_argument(
        '--name',
        type=str,
        default="result-img-num=25000_time-ps=50_bbox=(x=(0.000-6.231)_y=(0.590-6.821)_z=(3.392-9.623))_resolution=0.012461900",
    )

    args = parser.parse_args()

    path_to_extractor = args.path_to_extractor
    path_to_config = args.path_to_config
    default_path = args.default_path

    name = args.name

    pnm_path = join(default_path, "pnm")
    raw_img_path = join(default_path, "raw_images")
    euler_path = join(default_path, "euler.json")

    Path(pnm_path).mkdir(parents=True, exist_ok=True)

    # Opening JSON file
    with open(path_to_config) as f:
        # returns JSON object as
        # a dictionary
        jconfig = json.load(f)

    onlyfiles = [
        f
        for f in listdir(raw_img_path)
        if isfile(join(raw_img_path, f)) and '.raw' in f
    ]

    pattern = re.compile(
        r"result-img-num=(?P<step>\d+)"
        r"_time-ps=(?P<time_ps>\d+(?:\.\d+)?)"
        r"_bbox=\(x=\((?P<x_min>-?\d+(?:\.\d+)?)-(?P<x_max>-?\d+(?:\.\d+)?)\)"
        r"_y=\((?P<y_min>-?\d+(?:\.\d+)?)-(?P<y_max>-?\d+(?:\.\d+)?)\)"
        r"_z=\((?P<z_min>-?\d+(?:\.\d+)?)-(?P<z_max>-?\d+(?:\.\d+)?)\)\)"
        r"_resolution=(?P<resolution>\d+(?:\.\d+)?)"
    )

    for i, file in enumerate(onlyfiles):

        match = pattern.match(file)
        if not match:
            kprint("No match")
            continue
        data = match.groupdict()
        num = data["step"]
        x_min = data["x_min"]
        x_max = data["x_max"]
        y_min = data["y_min"]
        y_max = data["y_max"]
        z_min = data["z_min"]
        z_max = data["z_max"]
        resolution = float(data["resolution"])

        xs = int(round((float(x_max) - float(x_min)) / resolution))
        ys = int(round((float(y_max) - float(y_min)) / resolution))
        zs = int(round((float(z_max) - float(z_min)) / resolution))

        path_to_images = join(raw_img_path, file)
        path_to_pnm = join(pnm_path, file)

        angstrem_to_um = 0.0001

        pnm_pref = join(pnm_path, "pnm-" + file[11:-4])
        if isfile(pnm_pref + "_node1.dat"):
            kprint(f"Skip {num}")
            continue

        jconfig["input_data"]["filename"] = path_to_images
        jconfig["input_data"]["size"]["x"] = xs
        jconfig["input_data"]["size"]["y"] = ys
        jconfig["input_data"]["size"]["z"] = zs
        jconfig["output_data"]["statoil_prefix"] = pnm_pref
        jconfig["output_data"]["filename"] = pnm_pref
        jconfig["extraction_parameters"]["resolution"] = (
            resolution * angstrem_to_um
        )
        jconfig["extraction_parameters"]["length_unit_type"] = "UM"

        kprint(f"File name:{join(path_to_images)}")
        kprint(f"Size: {xs}, {ys}, {zs}")
        kprint(f"Output name: {pnm_pref}")
        kprint(f"Resolution: {resolution * 0.1} nm")

        with open(path_to_config, "w") as outfile:
            json.dump(jconfig, outfile)
        kprint(f"run: {path_to_extractor} {path_to_config}")

        process = subprocess.Popen(
            [
                path_to_extractor,
                path_to_config,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # wait for the process to terminate
        out, err = process.communicate()
        errcode = process.returncode
        if errcode != 0:
            print("Error!!!")
            break
        else:
            print(f"Sucssess: {pnm_pref}")
