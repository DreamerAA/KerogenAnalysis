from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor



class KeyPressInteractorStyle(vtkInteractorStyleTrackballCamera):
    def __init__(
        self,
        iren: vtkRenderWindowInteractor,
        status: bool = True,
    ):
        self.iren = iren
        self.status = status
        self.camera = (
            iren.GetRenderWindow()
            .GetRenderers()
            .GetFirstRenderer()
            .GetActiveCamera()
        )
        self.camera_default_position = self.camera.GetPosition()
        self.AddObserver('KeyPressEvent', self.key_press_event)

    def key_press_event(self, obj, event) -> None: # type: ignore
        key = self.iren.GetKeySym().lower()
        if key == 'q':
            self.status = False
        if key == 'r':
            self.camera.SetPosition(self.camera_default_position)
        return