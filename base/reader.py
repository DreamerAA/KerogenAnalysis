from typing import IO, Any, Tuple, List
import numpy.typing as npt
import numpy as np

from base.kerogendata import AtomData


def skip_line(file: IO, count: int = 1) -> None:  # type: ignore
    for _ in range(count):
        next(file)


class Reader:
    @staticmethod
    def read_struct_and_linked_list(
        path_to_structure: str, path_to_linked_list: str
    ) -> Tuple[Any, Any, Any]:
        atoms, size = Reader.read_raw_struct(path_to_structure)
        linked_list = Reader.read_linked_list(path_to_linked_list)
        return atoms, size, linked_list

    @staticmethod
    def read_linked_list(path_to_linked_list: str) -> List[Tuple[int, int]]:
        linked_list = []
        with open(path_to_linked_list) as f:
            for line in f:
                if "CONECT" in line:
                    n1, n2 = int(line[6:11]) - 1, int(line[11:16]) - 1
                    linked_list.append((n1, n2))
        return linked_list

    @staticmethod
    def read_psd(filename: str) -> npt.NDArray[np.float32]:
        with open(filename) as f:
            radiuses = [float(line.split("    ")[2]) for line in f]
        return np.array(radiuses, dtype=np.float32)

    @staticmethod
    def read_pnm_linklist(filename: str) -> npt.NDArray[np.int32]:
        with open(filename) as f:
            splited_lines = [line.split("    ") for line in f]
            n1 = [int(line[1]) for line in splited_lines]
            n2 = [int(line[2]) for line in splited_lines]
            dtp1 = [float(line[3]) for line in splited_lines]
            dtp2 = [float(line[4]) for line in splited_lines]
            length = [float(line[5]) for line in splited_lines]
            res = np.zeros(shape=(len(n1), 2), dtype=np.int32)
            res[:, 0] = n1
            res[:, 1] = n2

            res_l = np.zeros(shape=(len(dtp1), 3), dtype=np.float32)
            res_l[:, 0], res_l[:, 1], res_l[:, 2] = dtp1, dtp2, length
        return res, res_l

    @staticmethod
    def read_raw_struct(path_to_structure: str) -> Tuple[Any]:
        def type_to_type_id(type: str) -> int:
            if type[0] == 'c':
                return 0
            elif type[0] == 'o':
                return 1
            elif type[0] == 'n':
                return 2
            elif type[0] == 'h':
                return 3
            elif type[0] == 's':
                return 4
            return -1

        atoms = []
        methane = []
        with open(path_to_structure) as f:
            skip_line(f, 1)
            count_atoms = int(next(f))
            print(f" --- Count atoms: {count_atoms}")
            for i in range(count_atoms):
                line = next(f)
                try:
                    struct_number = int(line[0:5])
                    struct_type = line[5:8]

                    atom_id: str = str(line[8:15])
                    atom_id = atom_id.replace(" ", "")
                    type_id = type_to_type_id(atom_id)

                    x = float(line[20:28])
                    y = float(line[28:36])
                    z = float(line[36:44])

                    data = AtomData(
                        struct_number,
                        struct_type,
                        atom_id,
                        type_id,
                        np.array([x, y, z]),
                    )

                    if "CH4" in struct_type:
                        methane.append(data)
                    else:
                        atoms.append(data)
                except Exception as e:
                    print(i, line)
                    print(f"Error in line {i}: {line}, error: {e}")
            cell_sizes = next(f)

            str_size = list(filter(lambda x: x != '', cell_sizes.split(' ')))
            size = tuple([float(e) for e in str_size])
        atoms = np.array(atoms)
        return atoms, size
