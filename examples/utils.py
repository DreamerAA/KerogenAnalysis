import numpy as np
import numpy.typing as npt
from scipy.stats import poisson


def ps_generate(
    type: str, max_count_step: int = 100
) -> npt.NDArray[np.float32]:
    steps = np.arange(0, max_count_step)
    if type == 'poisson':
        steps = np.arange(0, max_count_step)
        prob = poisson.cdf(steps, 100, loc=-50)
        prob[:-1] = prob[:-1] - prob[0]
        prob[1:] = prob[1:] + (1 - prob[-1])

        ps = np.zeros(shape=(len(steps) + 1, 2), dtype=np.float32)
        ps[1:, 0] = steps
        ps[1:, 1] = prob
    else:
        steps = np.arange(0, max_count_step)
        prob = (steps.astype(np.float32)) * 0.01
        ps = np.zeros(shape=(len(steps) + 1, 2), dtype=np.float32)
        ps[1:, 0] = steps
        ps[1:, 1] = prob / prob[-1]
    return ps


def create_cdf(vals, n=30):
    p, bb = np.histogram(vals, bins=n)
    xdel = bb[1] - bb[0]
    x = (bb[:-1] + xdel * 0.5).reshape(n, 1)
    pn = (np.cumsum(p) / np.sum(p)).reshape(n, 1)
    pn[0] = 0
    return np.hstack((x, pn))


def write_binary_file(array: npt.NDArray[np.int8], file_name: str) -> None:
    with open(file_name, 'wb') as file:
        for i in range(array.shape[2]):
            for j in range(array.shape[1]):
                file.write(bytes(bytearray(array[:, j, i])))
