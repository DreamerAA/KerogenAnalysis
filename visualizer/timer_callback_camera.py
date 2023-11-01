from typing import List, Optional, Any


import numpy as np
import numpy.typing as npt
import math as m

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkRenderWindowInteractor,
)

def Rx(theta: float) -> npt.NDArray[np.float64]:
    return np.matrix(
        np.array(
            [
                [1, 0, 0],
                [0, m.cos(theta), -m.sin(theta)],
                [0, m.sin(theta), m.cos(theta)],
            ]
        )
    )


def Ry(theta: float) -> npt.NDArray[np.float64]:
    return np.matrix(
        np.array(
            [
                [m.cos(theta), 0, m.sin(theta)],
                [0, 1, 0],
                [-m.sin(theta), 0, m.cos(theta)],
            ]
        )
    )


def Rz(theta: float) -> npt.NDArray[np.float64]:
    return np.matrix(
        np.array(
            [
                [m.cos(theta), -m.sin(theta), 0],
                [m.sin(theta), m.cos(theta), 0],
                [0, 0, 1],
            ]
        )
    )


class vtkTimerCallbackCamera:
    def __init__(
        self,
        steps: int,
        actors: vtkActor,
        cameras: List[vtkCamera],
        iren: List[vtkRenderWindowInteractor],
    ):
        self.timer_count = 0
        self.steps = steps
        self.actors = actors
        self.cameras = cameras
        self.iren = iren
        self.timerId = None
        self.angle = 0.0
        self.cur_pos = np.array(cameras[0].GetPosition())

        self.astep = 0.15
        # a = self.astep*np.pi/180
        # self.Rxyz = Rz(a) * Ry(a) * Rx(a)

    def calcXYZ(self) -> npt.NDArray[np.float64]:
        a = self.angle * np.pi / 180
        self.Rxyz = Ry(a) * Rx(a)
        return (self.cur_pos * self.Rxyz).T.A.squeeze()  # type: ignore

    def execute(self, obj, event):  # type: ignore
        step = 0
        while step < self.steps:
            self.angle += self.astep
            cur_pos = self.calcXYZ()
            for camera in self.cameras:
                camera.SetPosition(cur_pos[0], cur_pos[1], cur_pos[2])
                # actor.RotateWXYZ(1, 0.2, 0.2, 0.2)#self.timer_count / 100.0, self.timer_count / 100.0, 0
            iren = obj
            iren.GetRenderWindow().Render()
            self.timer_count += 1
            step += 1
        if self.timerId:
            iren.DestroyTimer(self.timerId)

