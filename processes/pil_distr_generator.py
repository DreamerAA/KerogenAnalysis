from utils.types import NPFArray, f32
from scipy.spatial.distance import pdist
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed


class PiLDistrGenerator:
    def __init__(self, count_points: int = 10_000, seed: int = 123):
        self.count_points = count_points
        self.rng = np.random.default_rng(seed)

        # Кэш (если захочешь переиспользовать между вызовами)
        self._xyz_unit: np.ndarray | None = None
        self._d_unit_sorted: np.ndarray | None = None

    @staticmethod
    def upper_tri_masking(
        A: NPFArray,
    ) -> NPFArray:
        m = A.shape[0]
        r = np.arange(m)
        mask = r[:, None] < r
        return A[mask]

    @staticmethod
    def sample_distances_in_ball(
        xyz_unit: NPFArray,
        radius: float,
        k_pairs: int,
        rng: np.random.Generator,
    ) -> NPFArray:
        """
        xyz_unit: точки в единичном шаре (N,3)
        radius: масштаб
        k_pairs: сколько пар сэмплировать
        """
        xyz = (xyz_unit * radius).astype(np.float32, copy=False)
        n = xyz.shape[0]

        i = rng.integers(0, n, size=k_pairs)
        j = rng.integers(0, n, size=k_pairs)

        # чтобы не было i==j (нулевых расстояний)
        eq = i == j
        while np.any(eq):
            j[eq] = rng.integers(0, n, size=np.sum(eq))
            eq = i == j

        d = xyz[i] - xyz[j]
        return np.sqrt(np.sum(d * d, axis=1))

    @staticmethod
    def _make_xyz_in_unit_ball(
        count_points: int, rng: np.random.Generator
    ) -> NPFArray:
        xyz = rng.uniform(-1.0, 1.0, size=(count_points, 3)).astype(np.float32)
        # быстрее, чем sqrt
        xyz = xyz[np.sum(xyz * xyz, axis=1) < 1.0]
        return xyz

    @staticmethod
    def _iter_batches(
        arr: npt.NDArray[np.float64], batch_size: int
    ) -> Iterable[npt.NDArray[np.float64]]:
        for start in range(0, arr.size, batch_size):
            yield arr[start : start + batch_size]

    def gen_set(
        self,
        pore_radiuses: NPFArray,
        seed: int = 123,
        k_pairs: int = 100,
        n_jobs: int = 8,
        batch_size: int = 200,
        verbose_every_batches: int = 10,
    ):
        """
        Возвращает выборку расстояний (смесь по радиусам).
        - Для каждого радиуса генерируется k_pairs расстояний.
        - Радиусы обрабатываются батчами, батчи параллелятся.
        - RNG независимый на батч (через SeedSequence.spawn), воспроизводимо.
        """

        pore_radiuses = np.sort(pore_radiuses)
        # единый набор точек в единичном шаре — как у тебя раньше
        # (seed отдельный, чтобы не зависел от batch RNG)
        rng_xyz = np.random.default_rng(seed + 1_000_000)
        xyz_unit = self._make_xyz_in_unit_ball(self.count_points, rng_xyz)

        # батчи радиусов
        batches = list(self._iter_batches(pore_radiuses, batch_size))
        nb = len(batches)

        # независимые seed-ы на батчи
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(nb)

        def sim_batch(bi: int, rad_batch: NPFArray) -> NPFArray:
            rng = np.random.default_rng(child_seeds[bi])

            out = np.empty((rad_batch.size * k_pairs,), dtype=f32)
            off = 0
            for rad in rad_batch:
                d = PiLDistrGenerator.sample_distances_in_ball(
                    xyz_unit=xyz_unit,
                    radius=float(rad),
                    k_pairs=k_pairs,
                    rng=rng,
                )
                out[off : off + k_pairs] = d
                off += k_pairs

            if verbose_every_batches and (
                (bi + 1) % verbose_every_batches == 0 or (bi + 1) == nb
            ):
                print(f"Finish batch {bi+1}/{nb} (rads: {rad_batch.size})")

            return out.astype(f32, copy=False)

        pres: list[NPFArray] = Parallel(n_jobs=n_jobs)(
            delayed(sim_batch)(bi, rad_batch)
            for bi, rad_batch in enumerate(batches)
        )

        result = np.concatenate(pres)
        return result

    def _prepare_unit_distances(self) -> np.ndarray:
        """Генерируем точки в единичном шаре и считаем pdist один раз."""
        if self._d_unit_sorted is not None:
            return self._d_unit_sorted

        count_points = self.count_points

        # Генерация в кубе + отбрасывание вне шара (rejection sampling)
        xyz = self.rng.uniform(-1.0, 1.0, size=(count_points, 3)).astype(f32)
        dist = np.sqrt(np.sum(xyz * xyz, axis=1))
        xyz = xyz[dist < 1.0]

        self._xyz_unit = xyz

        # pdist быстрее и экономнее, чем pairwise_distances + upper_tri_masking
        d_unit = pdist(xyz, metric="euclidean").astype(
            f32
        )  # shape: (n*(n-1)/2,)
        d_unit.sort()  # сортируем один раз

        self._d_unit_sorted = d_unit
        return d_unit

    def get_curve(self, pore_radiuses: np.ndarray) -> np.ndarray:
        pore_radiuses = np.asarray(pore_radiuses, dtype=f32)
        pore_radiuses = np.sort(
            pore_radiuses
        )  # ВАЖНО: np.sort не сортирует inplace

        max_rad = float(pore_radiuses[-1])

        # Максимальная длина (как у тебя)
        max_length = np.sqrt(3.0 * ((1.5 * max_rad) ** 2))

        cl = 100
        nx_len = np.linspace(0.0, max_length, cl, dtype=f32)
        dl = float(
            nx_len[1] - nx_len[0]
        )  # ширина бина в СКАЛИРОВАННОМ пространстве

        # Один раз готовим отсортированные расстояния в единичном шаре
        d_unit_sorted = self._prepare_unit_distances()

        # Радиусы, которые реально считаем
        sample_rad = pore_radiuses[::10].astype(f32)

        def sim(i: int, radius: float) -> np.ndarray:
            # Бины для unit-расстояний: d_unit in [L/r, R/r]
            edges_unit = (nx_len / radius).astype(f32)

            # Индексы в отсортированном массиве
            idx = np.searchsorted(d_unit_sorted, edges_unit, side="right")
            counts = np.diff(idx).astype(f32)  # shape (cl-1,)

            # Нормировка как у тебя: интеграл pi(l) dl = 1
            norm = float(np.sum(counts) * dl)
            res = (counts / norm) if norm != 0.0 else counts
            return res

        # ВАЖНО: чтобы не копировать огромный d_unit_sorted в процессы,
        # используем потоки (общая память). searchsorted в numpy обычно отпускает GIL.
        pres = Parallel(n_jobs=8, prefer="threads")(
            delayed(sim)(i, float(rad)) for i, rad in enumerate(sample_rad)
        )

        pi_l_d = np.vstack(pres)  # (m, cl-1)
        pi_l = np.mean(pi_l_d, axis=0).astype(f32)  # (cl-1,)

        new_l = nx_len[:-1] + 0.5 * (nx_len[1:] - nx_len[:-1])

        pi_l_save = np.zeros((pi_l.shape[0], 2), dtype=f32)
        pi_l_save[:, 0] = new_l
        pi_l_save[:, 1] = pi_l
        return pi_l_save

    def get_conditional_curves(self, pore_radiuses: np.ndarray, step:int = 10):
        pore_radiuses = np.asarray(pore_radiuses, dtype=f32)
        pore_radiuses = np.sort(pore_radiuses)

        max_rad = float(pore_radiuses[-1])
        max_length = np.sqrt(3.0 * ((1.5 * max_rad) ** 2))

        cl = 100
        nx_len = np.linspace(0.0, max_length, cl, dtype=f32)
        dl = float(nx_len[1] - nx_len[0])

        d_unit_sorted = self._prepare_unit_distances()
        sample_rad = pore_radiuses[::step].astype(f32)

        def sim(radius: float) -> np.ndarray:
            edges_unit = (nx_len / radius).astype(f32)
            idx = np.searchsorted(d_unit_sorted, edges_unit, side="right")
            counts = np.diff(idx).astype(f32)
            norm = float(np.sum(counts) * dl)
            return (counts / norm) if norm != 0.0 else counts

        pi_l_cond = np.vstack([sim(float(r)) for r in sample_rad])
        l_centers = nx_len[:-1] + 0.5 * (nx_len[1:] - nx_len[:-1])

        return sample_rad, l_centers, pi_l_cond
