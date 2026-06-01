import sys
from os.path import isfile, realpath
from pathlib import Path

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from base.reader import StepsInfo


def run(data_path: str, save_path: str) -> StepsInfo:
    info = StepsInfo()
    fdata = open(data_path, 'r')
    if isfile(save_path):
        print(f"Skip trajectory:data_path={data_path}, save_path={save_path}")
        return info
    fsave = open(save_path, 'w')
    try:
        info = StepsInfo()
        for i, line in enumerate(fdata):
            if "Kerogen " in line:
                info.get_step(line)
            if "KRG" not in line:
                fsave.write(line)
    except Exception as e:
        print(f"Error in line {i}: {line}, error: {e}")
    return info


if __name__ == '__main__':
    for type in [2]:
        for temp in ["300", "400"]:
            for a in ["h2", "ch4"]:
                main_path = f"/media/andrey/Samsung_T5/PHD/Kerogen/type{type}matrix/{temp}K/{a}/"
                data_path = main_path + f"type{type}.{a}.{temp}.gro"
                save_path = main_path + "/trj.gro"
                run(data_path, save_path)
