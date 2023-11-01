import sys
import argparse
import json
import time
from pathlib import Path
from typing import Tuple, Any
from os import listdir
from os.path import isfile, join, dirname, realpath
import subprocess
from scipy.stats import weibull_min, exponweib
import numpy as np
import matplotlib.pyplot as plt

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import Reader

def extract_pnms():
    config_path = "/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/extract_config.json"
    path_to_images = "/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/images/"
    path_to_result = "/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/pnm/"
    # Opening JSON file
    f = open(config_path)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    onlyfiles = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    for i,file in enumerate(onlyfiles):
        start_time = time.time()
        resolution = float((file.split("=")[-1][:-6]))
        num = int((file.split("=")[1]).split("_")[0])
        if num == 20:
            continue
        full_path = join(path_to_images ,file)
        data["input_data"]["filename"] = full_path
        data["input_data"]["size"] = {"x": 500,"y": 500,"z": 500}
        data["output_data"]["statoil_prefix"] = join(path_to_result, f"num={num}")
        data["extraction_parameters"]["resolution"] = resolution*0.1

        with open(config_path, 'w') as f:
            json.dump(data, f)

        exe = "/home/andrey/DigitalCore/PNE/pore-network-extraction/build/bin/extractor_example"
        

        bashCommand = f"{exe} {config_path}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        process.wait()
        print(f" --- Elapsed time {time.time() - start_time} (s)")
        print(f" --- Process ending num = {num}, file {i+1} from {len(onlyfiles)}")
        


def extract_weibull_psd(path_to_pnm: str, scale: float, border: float = 0.02)->Tuple[Any]:
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=scale, border=border
    )
    psd_params = exponweib.fit(radiuses)
    tld_params = exponweib.fit(throat_lengths)
    return psd_params, tld_params, radiuses, throat_lengths


def build_distributions(path_to_pnms:str)->None:
    fig, axs = plt.subplots(1, 2)

    onlyfiles = [f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))]    
    onlyfiles = [file for file in onlyfiles if "_link1" in file]
    steps = [int((file.split("_")[0]).split("=")[1]) for file in onlyfiles]
    sorted_lfiles = list(zip(steps, onlyfiles))
    sorted_lfiles = sorted(sorted_lfiles, key=lambda x: x[0])
    
    for step, file in sorted_lfiles[::2]:
        psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd(
            join(path_to_pnms, file[:-10]), 1e10, 0.015
        )

        nx_rad = np.linspace(radiuses[0], radiuses[-1], 1000)        
        nx_len = np.linspace(throat_lengths[0], throat_lengths[-1], 1000)
        

        n = 50

        # axs[0].hist(radiuses,bins=n)
        # axs[1].hist(throat_lengths,bins=n)
        
        p, bb = np.histogram(radiuses, bins=n)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        axs[0].plot(x, pn, label=f"Histogram data step={step}")
        # axs[0].plot(nx_rad, fit_y_rad, label='Fitting')
        axs[0].set_title("PDF(Pore size distribution)")
        axs[0].set_xlabel("Pore diameter (A)")

        p, bb = np.histogram(throat_lengths, bins=n)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        axs[1].plot(x, pn, label=f"Histogram data step={step}")
        # axs[1].plot(nx_len, fit_y_len, label='Fitting')
        axs[1].set_title("PDF(Throat length distribution)")
        axs[1].set_xlabel("Throat length (A)")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # extract_pnms()
    build_distributions("/home/andrey/PHD/Kerogen/data/Kerogen/tmp/result_time_depend_struct/pnm/")
