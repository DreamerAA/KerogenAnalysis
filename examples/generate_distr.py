from typing import Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import exponweib
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from matplotlib import cm
import sys
from pathlib import Path
from os.path import realpath

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)


from base.reader import Reader



def extract_weibull_psd(path_to_pnm: str, scale: float, border: float = 0.02)->Tuple[Any]:
    radiuses, throat_lengths = Reader.read_pnm_data(
        path_to_pnm, scale=scale, border=border
    )
    psd_params = exponweib.fit(radiuses)
    tld_params = exponweib.fit(throat_lengths)
    return psd_params, tld_params, radiuses, throat_lengths


def generate_distribution(path_to_pnm:str, path_to_save:str) -> None:
    psd_params, tld_params, radiuses, throat_lengths = extract_weibull_psd(
        path_to_pnm, 1e10, 0.015
    )

    max_rad = radiuses[-1]
    max_length = np.sqrt(3 * ((1.5 * max_rad) ** 2))

    cl = 100
    nx_len = np.linspace(0, max_length, cl)

    dl = nx_len[1] - nx_len[0]

    count_points = 10_000
    xyz = np.zeros(shape=(count_points, 3), dtype=np.float32)
    xyz[:, 0] = np.random.uniform(-1, 1, size=count_points)
    xyz[:, 1] = np.random.uniform(-1, 1, size=count_points)
    xyz[:, 2] = np.random.uniform(-1, 1, size=count_points)
    dist = np.sqrt(np.sum(xyz**2, axis=1))
    xyz = xyz[dist < 1, :]

    def upper_tri_masking(
        A: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        m = A.shape[0]
        r = np.arange(m)
        mask = r[:, None] < r
        return A[mask]

    def sim(num: int, m: int, radius: float) -> npt.NDArray[np.float32]:
        len_distr = np.zeros(shape=(cl - 1,), dtype=np.int32)
        xyz_scaled = xyz * radius
        distances = pairwise_distances(xyz_scaled, xyz_scaled, n_jobs=4)
        distances = upper_tri_masking(distances)

        for i in range(1, cl):
            ll, rl = nx_len[(i - 1) : (i + 1)]
            len_distr[i - 1] = np.sum(
                np.logical_and(distances > ll, distances <= rl)
            )
        print(f" --- Result of {num+1} from {m}")
        return len_distr / np.sum(len_distr * dl)

    def PiLD_3D():
        cr = 50
        nx_rad = np.linspace(0, max_rad, cr)

        pi_l_d = np.zeros(shape=(cr - 1, cl - 1))

        pres = Parallel(n_jobs=10)(
            delayed(sim)(i, len(nx_rad[1:]), rad)
            for i, rad in enumerate(nx_rad[1:])
        )
        for i, res in enumerate(pres):
            pi_l_d[i, :] = res

        new_l = nx_len[:-1] + (nx_len[1:] - nx_len[:-1]) * 0.5

        X = np.copy(new_l)
        Y = nx_rad[1:]
        X, Y = np.meshgrid(X, Y)
        Z = np.copy(pi_l_d)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel("Segemnt length (nm)")
        ax.set_ylabel("Domain radius (nm)")
        ax.set_zlabel("Count")

    sample_rad = radiuses[::30]
    pi_l_d = np.zeros(shape=(sample_rad.shape[0], cl - 1))

    pres = Parallel(n_jobs=10)(
        delayed(sim)(i, len(sample_rad), rad)
        for i, rad in enumerate(sample_rad)
    )

    for i, res in enumerate(pres):
        pi_l_d[i, :] = res

    pi_l = np.mean(pi_l_d, axis=0)
    new_l = nx_len[:-1] + (nx_len[1:] - nx_len[:-1]) * 0.5

    p, bb = np.histogram(throat_lengths, bins=50)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + xdel * 0.5
    pn = p / np.sum(p * xdel)

    pi_l_save = np.zeros(shape=(pi_l.shape[0], 2), dtype=np.float32)
    pi_l_save[:, 1] = pi_l
    pi_l_save[:, 0] = new_l
    path_to_save
    np.save(path_to_save + "_pi_l.npy", pi_l_save)
    np.save(path_to_save + "_throat_lengths.npy", throat_lengths)


if __name__ == '__main__':
    path = "/home/andrey/PHD/Kerogen/data/Kerogen/time_trapping_results/ch4/"
    generate_distribution(path + "num=1597500000_500_500_500",
                          path + "num=1597500000_500_500_500")