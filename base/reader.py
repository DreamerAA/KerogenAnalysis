from typing import IO, Any, Tuple, List
import numpy.typing as npt
import numpy as np
from io import TextIOWrapper
from base.kerogendata import AtomData


def skip_line(file: IO, count: int = 1) -> bool:  # type: ignore
    try:
        for _ in range(count):
            next(file)
        return False
    except StopIteration:
        return True


class StepsInfo:
    def __init__(self):
        self.steps = []
        self.delta = -1

    def getStep(self, line: str) -> None:
        els = (line[:-1]).split('=')
        res = int(els[2])
        if res < 0:
            res = self.steps[-1] + self.delta

        self.steps.append(res)

        if len(self.steps) == 2:
            self.delta = self.steps[-1] - self.steps[-2]
        elif len(self.steps) > 2:
            assert self.delta == (self.steps[-1] - self.steps[-2])


class Reader:
    @staticmethod
    def read_structures_by_num(
        path_to_structure: str, indexes: List[int]
    ) -> List[Tuple[List[AtomData], Tuple[float, float, float]]]:
        structures = []
        info = StepsInfo()
        with open(path_to_structure) as f:
            is_end = False
            while not is_end:
                num = Reader.read_head_struct(f, info)
                # print(f" -- Current num: {num}")
                if num == -1:
                    is_end = True
                    continue

                if num not in indexes:
                    is_end = Reader.skip_struct_main_part(f)
                    if is_end:
                        break
                else:
                    atoms, size = Reader.read_raw_struct_ff_main(f)
                    structures.append((num, np.array(atoms), size))
                    print(" -- Reading struct is ended!")

        return structures

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
    def read_pnm_linklist(filename: str) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        count = -1
        with open(filename) as f:
            if count == -1:
                count = int(next(f))
            splited_lines = f.readlines()

            def extract_num(line: str, ind: int):
                splited_line = line.split(' ')
                splited_line = [
                    x for x in splited_line if len(x) > 0 and x != ''
                ]
                return splited_line[ind]

            n1 = [int(extract_num(line, 1)) for line in splited_lines]
            n2 = [int(extract_num(line, 2)) for line in splited_lines]
            rad = [float(extract_num(line, 3)) for line in splited_lines]
            shape_factor = [
                float(extract_num(line, 4)) for line in splited_lines
            ]
            length = [float(extract_num(line, 5)) for line in splited_lines]
            res = np.zeros(shape=(len(n1), 2), dtype=np.int32)
            res[:, 0] = n1
            res[:, 1] = n2

            res_l = np.zeros(shape=(len(rad), 3), dtype=np.float32)
            res_l[:, 0], res_l[:, 1], res_l[:, 2] = rad, shape_factor, length
        return res, res_l

    @staticmethod
    def read_link_radiuses(filename: str) -> npt.NDArray[np.float32]:
        with open(filename) as f:
            radiuses = [float(line.split("    ")[2]) for line in f]
        return np.array(radiuses, dtype=np.float32)

    @staticmethod
    def type_to_type_id(type: str) -> int:
        if type[0].lower() == 'c':
            return 0
        elif type[0].lower() == 'o':
            return 1
        elif type[0].lower() == 'n':
            return 2
        elif type[0].lower() == 'h':
            return 3
        elif type[0].lower() == 's':
            return 4
        return -1

    @staticmethod
    def read_raw_struct(path_to_structure: str) -> Tuple[Any]:
        with open(path_to_structure) as f:
            atoms, size, _ = Reader.read_raw_struct_ff(f)
        atoms = np.array(atoms)
        return atoms, size

    @staticmethod
    def read_head_struct(f, info: StepsInfo = StepsInfo()) -> int:
        try:
            simul_num = str(next(f))
            info.getStep(simul_num)
        except StopIteration:
            return -1
        return info.steps[-1]
        

    @staticmethod
    def read_raw_struct_ff(
        f: TextIOWrapper,
    ) -> Tuple[List[AtomData], Tuple[float, float, float], int]:
        simul_num = Reader.read_head_struct(f)
        if simul_num == -1:
            return None, None, simul_num
        atoms, size = Reader.read_raw_struct_ff_main(f)
        return atoms, size, simul_num

    @staticmethod
    def read_raw_struct_ff_main(
        f: TextIOWrapper,
    ) -> Tuple[List[AtomData], Tuple[float, float, float]]:
        atoms = []

        count_atoms = int(next(f))
        print(f" --- Count atoms: {count_atoms}")
        for i in range(count_atoms):
            line = next(f)
            try:
                struct_number = int(line[0:5])
                struct_type = line[5:8]

                atom_id: str = str(line[8:15])
                atom_id = atom_id.replace(" ", "")
                type_id = Reader.type_to_type_id(atom_id)
                if type_id == -1:
                    print(f"Error in line {i}: {line}, error: {e}")
                    raise RuntimeError()
                
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

                if "KRG" in struct_type:
                    atoms.append(data)
                    
            except Exception as e:
                print(i, line)
                print(f"Error in line {i}: {line}, error: {e}")
        cell_sizes = next(f)

        str_size = list(filter(lambda x: x != '', cell_sizes.split(' ')))
        size = tuple([float(e) for e in str_size])
        atoms = np.array(atoms)
        return atoms, size

    @staticmethod
    def skip_struct(f: TextIOWrapper) -> bool:
        is_end = skip_line(f, 1)
        if is_end:
            return is_end
        count_atoms = int(next(f))
        is_end = skip_line(f, count_atoms + 1)
        return is_end

    def skip_struct_main_part(f: TextIOWrapper) -> bool:
        count_atoms = int(next(f))
        is_end = skip_line(f, count_atoms + 1)
        return is_end

    @staticmethod
    def read_pnm_data(
        path_to_pnm: str, scale: float, border: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        path_to_node_2 = path_to_pnm + "_node2.dat"
        path_to_link_1 = path_to_pnm + "_link1.dat"

        radiuses = Reader.read_psd(path_to_node_2)
        radiuses *= scale

        linked_list, t_throat_lengths = Reader.read_pnm_linklist(path_to_link_1)
        mask0 = linked_list[:, 0] <= 0
        mask1 = linked_list[:, 1] <= 0
        nn1 = linked_list[mask0, 1] - 1
        nn0 = linked_list[mask1, 0] - 1

        node_mask = np.ones(shape=(len(radiuses),), dtype=np.bool_)
        node_mask[nn0] = False
        node_mask[nn1] = False

        radiuses = radiuses[node_mask]
        radiuses.sort()
        radiuses = radiuses[radiuses > border]

        t_throat_lengths = t_throat_lengths[~np.logical_or(mask0, mask1), :]
        throat_lengths = t_throat_lengths[:, 2]
        throat_lengths *= scale
        throat_lengths.sort()
        return radiuses, throat_lengths

    @staticmethod
    def read_pore_positions(path: str) -> np.ndarray:
        with open(path, "rt") as f:
            line = f.readline()
            count_pores = int(line.split("    ")[0])
            positions = np.zeros((count_pores, 3), dtype=np.float32)
            for i in range(count_pores):
                line = f.readline()
                splits = line.split("    ")
                x = float(splits[1])
                y = float(splits[2])
                z = float(splits[3])
                positions[i, :] = [x, y, z]
        return positions


    @staticmethod
    def read_pnm_ext_data(
        path_to_pnm: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        path_to_node_1 = path_to_pnm + "_node1.dat"
        path_to_node_2 = path_to_pnm + "_node2.dat"
        path_to_link_1 = path_to_pnm + "_link1.dat"

        radiuses = Reader.read_psd(path_to_node_2)

        linked_list, t_throat_lengths = Reader.read_pnm_linklist(path_to_link_1)
        mask0 = linked_list[:, 0] <= 0
        mask1 = linked_list[:, 1] <= 0


        throat_mask = np.logical_not(np.logical_or(mask0, mask1))
        t_throat_lengths = t_throat_lengths[throat_mask, :]
        linked_list = linked_list[throat_mask]
        linked_list[:, 0] -= 1
        linked_list[:, 1] -= 1

        return radiuses, t_throat_lengths, linked_list, Reader.read_pore_positions(path_to_node_1)

    @staticmethod
    def read_np_img(path: str) -> npt.NDArray[np.int32]:
        return np.load(path)
