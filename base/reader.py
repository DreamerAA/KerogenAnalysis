import re
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
        self.pattern = re.compile(
            r"t=\s*([0-9]+(?:\.[0-9]+)?)\s+step=\s*([0-9]+)"
        )
        self.steps = []
        self.times = []  # ps
        self.delta = -1

    def get_step(self, line: str) -> None:
        match = self.pattern.search(line)
        assert match

        t = int(float(match.group(1)))
        step = int(match.group(2))
        self.steps.append(step)
        self.times.append(t)


_TYPE_MAP = {'c': 0, 'o': 1, 'n': 2, 'h': 3, 's': 4}


class Reader:
    @staticmethod
    def read_structures_by_num(
        path_to_structure: str, indexes: List[int]
    ) -> List[Tuple[int, int, List[AtomData], Tuple[float, float, float]]]:
        structures = []
        info = StepsInfo()
        indexes_set = set(indexes)
        with open(path_to_structure) as f:
            is_end = False
            while not is_end:
                num, time = Reader.read_head_struct(f, info)
                if num == -1:
                    break

                if num not in indexes_set:
                    is_end = Reader.skip_struct_main_part(f)
                else:
                    atoms, size = Reader.read_raw_struct_ff_main(f)
                    structures.append((num, time, atoms, size))

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
    def read_pnm_linklist(
        filename: str,
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
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
        return _TYPE_MAP.get(type[0].lower(), -1)

    @staticmethod
    def read_raw_struct(path_to_structure: str) -> Tuple[Any]:
        with open(path_to_structure) as f:
            atoms, size, _ = Reader.read_raw_struct_ff(f)
        atoms = np.array(atoms)
        return atoms, size

    @staticmethod
    def read_head_struct(f, info: StepsInfo = StepsInfo()) -> Tuple[int, int]:
        try:
            simul_num = str(next(f))
            info.get_step(simul_num)
        except StopIteration:
            return -1, -1
        return info.steps[-1], info.times[-1]

    @staticmethod
    def read_raw_struct_ff(
        f: TextIOWrapper,
    ) -> Tuple[List[AtomData], Tuple[float, float, float], int]:
        simul_num, time = Reader.read_head_struct(f)
        if simul_num == -1:
            return None, None, simul_num
        atoms, size = Reader.read_raw_struct_ff_main(f)
        return atoms, size, simul_num

    @staticmethod
    def read_raw_struct_ff_main(
        f: TextIOWrapper,
    ) -> Tuple[List[AtomData], Tuple[float, float, float]]:
        count_atoms = int(next(f))
        lines = [next(f) for _ in range(count_atoms)]

        krg_lines = [l for l in lines if l[5:8] == 'KRG']
        n = len(krg_lines)

        coords = np.empty((n, 3), dtype=np.float32)
        struct_numbers = np.empty(n, dtype=np.int32)
        struct_types: List[str] = []
        atom_ids: List[str] = []
        type_ids = np.empty(n, dtype=np.int8)

        for i, l in enumerate(krg_lines):
            struct_numbers[i] = int(l[0:5])
            struct_types.append(l[5:8])
            aid = l[8:15].strip()
            atom_ids.append(aid)
            type_ids[i] = Reader.type_to_type_id(aid)
            coords[i, 0] = float(l[20:28])
            coords[i, 1] = float(l[28:36])
            coords[i, 2] = float(l[36:44])

        atoms = np.array(
            [
                AtomData(
                    int(struct_numbers[i]),
                    struct_types[i],
                    atom_ids[i],
                    int(type_ids[i]),
                    coords[i],
                )
                for i in range(n)
            ]
        )

        cell_sizes = next(f)
        size = tuple(float(x) for x in cell_sizes.split() if x)
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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        path_to_pnm: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        return (
            radiuses,
            t_throat_lengths,
            linked_list,
            Reader.read_pore_positions(path_to_node_1),
        )

    @staticmethod
    def read_np_img(path: str) -> npt.NDArray[np.int32]:
        return np.load(path)
