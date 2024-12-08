import argparse
import json
import subprocess
from os import listdir
from os.path import dirname, isfile, join, realpath
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_extractor',
        type=str,
        default="/home/andrey/DigitalCore/PNE/pore-network-extraction/build/clang-15-release-cpu/bin/extractor_example",
    )

    default_path = "/media/andrey/Samsung_T5/PHD/Kerogen/type2matrix/300K/h2/"

    parser.add_argument(
        '--path_to_pnm_folder',
        type=str,
        # default="/media/andrey/Samsung_T5/PHD/Kerogen/400K/h2/pnm/",
        default=default_path + "/pnm/",
    )

    parser.add_argument(
        '--path_to_img_folder',
        type=str,
        # default="/media/andrey/Samsung_T5/PHD/Kerogen/400K/h2/images/",
        default=default_path + "/images/",
    )

    parser.add_argument(
        '--path_to_config',
        type=str,
        default="/home/andrey/DigitalCore/PNE/pore-network-extraction/example/config/ExtractorExampleConfig.json",
    )
    parser.add_argument(
        '--path_to_save_euler',
        type=str,
        # default='/media/andrey/Samsung_T5/PHD/Kerogen/400K/h2/euler.json',
        default=default_path + "/euler.json",
    )

    args = parser.parse_args()

    path_to_images = args.path_to_img_folder
    path_to_pnm = args.path_to_pnm_folder
    path_to_extractor = args.path_to_extractor
    path_to_config = args.path_to_config
    path_to_save_euler = args.path_to_save_euler

    Path(path_to_pnm).mkdir(parents=True, exist_ok=True)

    # Opening JSON file
    with open(path_to_config) as f:
        # returns JSON object as
        # a dictionary
        jconfig = json.load(f)

    onlyfiles = [
        f
        for f in listdir(path_to_images)
        if isfile(join(path_to_images, f)) and '.raw' in f
    ]

    output_save = {}
    if isfile(path_to_save_euler):
        with open(path_to_save_euler, "r") as f:
            output_save = json.load(f)

    for i, file in enumerate(onlyfiles):
        lfile = file.split('_')
        parts = [
            l.split('=')[1]
            for l in lfile
            if "num" in l or "resolution" in l or "is=" in l
        ]
        print(parts)
        num = int(parts[0])
        xs, ys, zs = [int(e) for e in parts[1][1:-1].split(',')]
        resolution = float(parts[2][:-5])

        pnm_pref = join(path_to_pnm, f"num={num}_{xs}_{ys}_{zs}")
        jconfig["input_data"]["filename"] = join(path_to_images, file)
        jconfig["input_data"]["size"]["x"] = xs
        jconfig["input_data"]["size"]["y"] = ys
        jconfig["input_data"]["size"]["z"] = zs
        jconfig["output_data"]["statoil_prefix"] = pnm_pref
        jconfig["output_data"]["filename"] = pnm_pref
        jconfig["extraction_parameters"]["resolution"] = resolution * 0.0001
        jconfig["extraction_parameters"]["length_unit_type"] = "UM"

        if isfile(pnm_pref + "_node1.dat") and parts[0] in output_save:
            print(f"Skip {num}")
            continue

        print(f"--- File name:{join(path_to_images,file)}")
        print(f"--- Size: {xs}, {ys}, {zs}")
        print(f"--- Output name: {pnm_pref}")
        print(f"--- Resolution: {resolution*0.1} nm")

        with open(path_to_config, "w") as outfile:
            json.dump(jconfig, outfile)

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
        euler = int(out.decode("utf-8").splitlines()[0].split(':')[1])
        output_save[num] = euler
        errcode = process.returncode
        if errcode != 0:
            print("Error!!!")
            break
        else:
            print(f"Sucssess: {output_save[num]}")
            print(f"go next to {i+1} from {len(onlyfiles)}")

        with open(path_to_save_euler, 'w') as f:
            json.dump(output_save, f)
