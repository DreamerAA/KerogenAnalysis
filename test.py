
prefix = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"

with open(prefix + "type1.ch4.1.gro", 'rt') as file:
    with open("../data/part.gro", 'wt') as res_file:
        for _ in range(70000):
            line = file.readline()
            if not line:
                break
            res_file.write(line)