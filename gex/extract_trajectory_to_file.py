from os.path import isfile


def run(data_path: str, save_path: str):
    fdata = open(data_path, 'r')
    if isfile(save_path):
        print(f"Skip trajectory:data_path={data_path}, save_path={save_path}")
        return
    fsave = open(save_path, 'w')
    try:
        for i, line in enumerate(fdata):
            if "KRG" not in line:
                fsave.write(line)
    except Exception as e:
        print(f"Error in line {i}: {line}, error: {e}")
        return


if __name__ == '__main__':
    main_path = "/media/andrey/Samsung_T5/PHD/Kerogen/"
    for temp in ["300K/", "400K/"]:
        for a in ["h2", "ch4"]:
            data_path = main_path + temp + f"{a}/type1.{a}.1.gro"
            save_path = main_path + temp + a + "/trj.gro"
            run(data_path, save_path)
