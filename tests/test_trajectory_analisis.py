import numpy as np
import numpy.typing as npt
import pytest

from base.boundingbox import BoundingBox
from base.trajectory import Trajectory
from processes.trajectory_analyzer import SpectralAnalizer
import os


@pytest.fixture
def test_path() -> str:
    return "/".join(os.path.abspath(__file__).split("/")[:-1]) + "/"


@pytest.fixture
def trajectory(test_path: str) -> Trajectory:
    fpoints = np.load(test_path + "fpoints.npy")
    times = np.array(range(fpoints.shape[0]), dtype=np.float64) * 100
    bbox = BoundingBox()
    bbox.update(np.array([0, 0, 0]))
    bbox.update(np.array([4, 4, 4]))
    trj = Trajectory(fpoints, times, bbox, atom_size=0.9)
    return trj


@pytest.fixture
def expected_result(test_path: str) -> npt.NDArray[np.bool_]:
    return np.load(test_path + "expected_result.npy")  # type: ignore


def test_trajectory_analizer_regression(trajectory: Trajectory, expected_result: npt.NDArray[np.bool_], test_path: str) -> None:
    _ = SpectralAnalizer(trajectory)
    print(trajectory.traps)
    print("trajectory.traps")
    assert trajectory.traps is not None
    assert trajectory.traps.size > 0
    assert np.all(trajectory.traps == expected_result)
