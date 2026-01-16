import numpy as np
import numpy.typing as npt
import pytest
from base.boundingbox import BoundingBox
from base.trajectory import Trajectory
from processes.trajectory_analyzer import TrajectoryAnalizer, AnalizerParams
import os
import pickle
import scipy


@pytest.fixture
def test_path() -> str:
    return "/".join(os.path.abspath(__file__).split("/")[:-1]) + "/"


@pytest.fixture
def trajectory(test_path: str) -> Trajectory:
    return pickle.load(open(test_path + "trajectory.pickle", "rb"))  # type: ignore


@pytest.fixture
def expected_result(test_path: str) -> npt.NDArray[np.bool_]:
    matlab_res = scipy.io.loadmat(test_path + "result_list_trapped.mat")
    return matlab_res["list_trapped"][0]  # type: ignore


def test_trajectory_analizer_regression(trajectory: Trajectory, expected_result: npt.NDArray[np.bool_]) -> None:
    _ = TrajectoryAnalizer(trajectory, AnalizerParams())
    print(trajectory.traps)
    print("trajectory.traps")
    assert trajectory.traps is not None
    assert trajectory.traps.size > 0
    assert np.all(trajectory.traps == expected_result)


@pytest.fixture
def trajectory2(test_path: str) -> Trajectory:
    fpoints = np.load(test_path + "fpoints.npy")
    times = np.array(range(fpoints.shape[0]), dtype=np.float32) * 100
    bbox = BoundingBox()
    bbox.update(np.array([0, 0, 0]))
    bbox.update(np.array([4, 4, 4]))
    trj = Trajectory(fpoints, times, bbox, atom_size=0.9)
    return trj


@pytest.fixture
def expected_result2(test_path: str) -> npt.NDArray[np.bool_]:
    return np.load(test_path + "expected_result.npy")


def test2_trajectory_analizer_regression(trajectory2: Trajectory, expected_result2: npt.NDArray[np.bool_]) -> None:
    _ = TrajectoryAnalizer(trajectory2, AnalizerParams())
    print(trajectory2.traps)
    assert trajectory2.traps is not None
    assert trajectory2.traps.size > 0
    assert np.all(trajectory2.traps == expected_result2)
