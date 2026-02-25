import numpy as np
import numpy.typing as npt
from scipy import stats
from typing import List, Any, TextIO
from base.kerogendata import AtomData
import builtins
import sys
from scipy.stats import poisson
from base.boundingbox import BoundingBox
from utils.types import NPFArray, NPIArray, NPBArray


def create_empirical_cdf(vals, n=30):
    p, bb = np.histogram(vals, bins=n)
    xdel = bb[1] - bb[0]
    x = (bb[:-1] + xdel * 0.5).reshape(n, 1)
    pn = (np.cumsum(p) / np.sum(p)).reshape(n, 1)
    pn[0] = 0
    return np.hstack((x, pn))


import numpy as np
import numpy.typing as npt
from scipy.stats import poisson


def ps_generate(
    type: str, max_count_step: int = 100
) -> npt.NDArray[np.float32]:
    steps = np.arange(0, max_count_step, dtype=np.float32)

    ps = np.zeros((len(steps), 2), dtype=np.float32)
    ps[:, 0] = steps

    if type == "poisson":
        # ВАЖНО: loc=-50 сдвигает распределение. это может давать "странные" формы на малом диапазоне steps.
        raw = poisson.cdf(steps.astype(int), 100, loc=-50).astype(np.float32)

        denom = raw[-1] - raw[0]
        if denom <= 0:
            # на этом диапазоне CDF почти не растёт -> деградируем в ступеньку
            prob = np.zeros_like(raw)
            prob[-1] = 1.0
        else:
            prob = (raw - raw[0]) / denom
            prob[0] = 0.0
            prob[-1] = 1.0
            prob = np.maximum.accumulate(prob)

        ps[:, 1] = prob

    else:
        # "uniform" ветка: линейная CDF
        prob = steps / steps[-1] if steps[-1] != 0 else np.zeros_like(steps)
        prob[0] = 0.0
        prob[-1] = 1.0
        ps[:, 1] = prob.astype(np.float32)

    return ps


def write_binary_file(array: npt.NDArray[np.int8], file_name: str) -> None:
    with open(file_name, 'wb') as file:
        for i in range(array.shape[2]):
            for j in range(array.shape[1]):
                file.write(bytes(bytearray(array[:, j, i])))


def kprint(
    *args: Any,
    sep: str | None = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
    prefix: str = " --- ",
) -> None:
    """
    Полный аналог print(...) с префиксом перед сообщением.
    - *args: любые объекты как и в print
    - sep: разделитель между args; если None — без разделителя (как в print)
    - end: окончание строки
    - file: поток вывода (по умолчанию sys.stdout)
    - flush: принудительная очистка буфера
    - prefix: настраиваемый префикс (по умолчанию ' --- ')
    """
    if file is None:
        file = sys.stdout

    joiner = "" if sep is None else sep
    body = joiner.join(map(str, args))  # как print: str() для каждого аргумента
    builtins.print(prefix + body, end=end, file=file, flush=flush)


def create_box_mask(atoms: List[AtomData], box: BoundingBox):
    removed_atoms = set()
    rm_mask = np.array(range(len(atoms)), dtype=np.bool_)
    for i, a in enumerate(atoms):
        rm_mask[i] = box.is_inside(a.pos)
        if ~rm_mask[i]:
            removed_atoms.add(i)
    return removed_atoms, rm_mask


def point_generation() -> npt.NDArray[np.float32]:
    def gen_point(
        xc: float,
        yc: float,
        zc: float,
        count: int,
        xstd: float = 0.2,
        ystd: float = 0.2,
        zstd: float = 0.2,
    ) -> npt.NDArray[np.float32]:
        points = np.zeros(shape=(count, 3), dtype=np.float32)
        points[:, 0] = stats.norm.rvs(xc, xstd, size=count)
        points[:, 1] = stats.norm.rvs(yc, ystd, size=count)
        points[:, 2] = stats.norm.rvs(zc, zstd, size=count)
        return points

    x1, y1 = 0.5, 0.5
    x2, y2 = 3.5, 3.5
    count = 200
    points1 = gen_point(x1, y1, 1.0, count)
    p12 = np.array(
        [
            x1,
            y1 + (y2 - y1) / 4,
            1.0,
            x1,
            y1 + (y2 - y1) * 2 / 4,
            1.0,
            x1,
            y1 + (y2 - y1) * 3 / 4,
            1.0,
        ],
        dtype=np.float32,
    ).reshape(3, 3)
    points2 = gen_point(x1, y2, 1.5, count)
    p23 = np.array(
        [
            x1 + (x2 - x1) / 4,
            y2,
            1.0,
            x1 + (x2 - x1) * 2 / 4,
            y2,
            1.0,
            x1 + (x2 - x1) * 3 / 4,
            y2,
            1.0,
        ],
        dtype=np.float32,
    ).reshape(3, 3)
    points3 = gen_point(x2, y2, 2.0, count)
    p34 = np.array(
        [
            x2,
            y1 + (y2 - y1) * 3 / 4,
            1.0,
            x2,
            y1 + (y2 - y1) * 2 / 4,
            1.0,
            x2,
            y1 + (y2 - y1) / 4,
            1.0,
        ],
        dtype=np.float32,
    ).reshape(3, 3)
    points4 = gen_point(x2, y1, 2.5, count)
    return np.vstack((points1, p12, points2, p23, points3, p34, points4))
