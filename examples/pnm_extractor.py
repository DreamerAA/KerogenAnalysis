import subprocess
from os import listdir
from os.path import isfile, join, dirname, realpath
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_extractor',
        type=str,
        default="../data/Kerogen/traj.gro",
    )
    parser.add_argument(
        '--path_to_pnm_folder',
        type=str,
        default="/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/pnm/",
    )

    parser.add_argument(
        '--path_to_img_folder',
        type=str,
        default="/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/img_to_pnm/",
    )

    parser.add_argument(
        '--path_to_config',
        type=str,
        default="/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/extract_config.json",
    )

    args = parser.parse_args()

    path_to_images = args.path_to_img_folder
    path_to_pnm = args.path_to_pnm_folder
    path_to_extractor = args.path_to_extractor
    path_to_config = args.path_to_config

    # Opening JSON file
    with open(path_to_config) as f:
        # returns JSON object as 
        # a dictionary
        jconfig = json.load(f)
        
    onlyfiles = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]

    for i, file in enumerate(onlyfiles):
        lfile = file.split('_')
        parts = [l.split('=')[1] for l  in lfile if "num" in l or "resolution" in l or "is=" in l]
        print(parts)
        num = int(parts[0])
        xs, ys, zs = [int(e) for e in parts[1][1:-1].split(',')]
        resolution = float(parts[2][:-5])

        pnm_pref = join(path_to_pnm, f"num={num}_{xs}_{ys}_{zs}")
        jconfig["input_data"]["filename"] = join(path_to_images,file)
        jconfig["input_data"]["size"]["x"] = xs
        jconfig["input_data"]["size"]["y"] = ys
        jconfig["input_data"]["size"]["z"] = zs
        jconfig["output_data"]["statoil_prefix"] = pnm_pref
        jconfig["output_data"]["filename"] = pnm_pref
        jconfig["extraction_parameters"]["resolution"] = resolution*0.1

        with open(path_to_config, "w") as outfile:
            json.dump(jconfig, outfile)

        process = subprocess.Popen(["/home/andrey/DigitalCore/PNE/pore-network-extraction/build/bin/extractor_example", path_to_config], 
                                    
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
        # wait for the process to terminate
        out, err = process.communicate()
        errcode = process.returncode
        if errcode != 0:
            print("Error!!!")
            break
        else:
            print(f"go next to {i+1} from {len(onlyfiles)}")
        
