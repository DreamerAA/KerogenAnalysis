from vtkmodules.vtkRenderingCore import (
    vtkRenderWindowInteractor,
)
import sys
from pathlib import Path
from os.path import realpath

path = Path(realpath(__file__))
parent_dir = str(path.parent.parent.absolute())
sys.path.append(parent_dir)

from visualizer.interactor_styles import KeyPressInteractorStyle

class WinStructCollection:
    def __init__(self, interactor: vtkRenderWindowInteractor):
        self.interactor = interactor
        self.renWin = self.interactor.GetRenderWindow()
        self.kpis = KeyPressInteractorStyle(iren=self.interactor)
        self.interactor.SetInteractorStyle(self.kpis)
        self.running = True

    def clear(self)->None:
        self.renWin.Finalize()
        del self.renWin, self.interactor
