import math as m
import random
from dataclasses import dataclass
from typing import List, Optional, Any

import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import vtk
from skimage import measure
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkLine
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkFiltersSources import vtkLineSource, vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

from base.boundingbox import BoundingBox
from base.trajectory import Trajectory


class KeyPressInteractorStyle(vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None, status=True):
        self.parent = vtkRenderWindowInteractor()
        self.status = status
        if parent is not None:
            self.parent = parent

        self.AddObserver('KeyPressEvent', self.key_press_event)

    def key_press_event(self, obj, event):
        key = self.parent.GetKeySym().lower()
        if key == 'e' or key == 'q':
            self.status = False
        return


interactors: List[vtkRenderWindowInteractor] = []
kpis: List[KeyPressInteractorStyle] = []
running: List[bool] = []


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


@dataclass
class AnimationActorData:
    sphere_actor: vtkActor
    # polyline_actor: vtkActor
    points: npt.NDArray[np.float32]
    # dists: npt.NDArray[np.float32]
    # polydata: vtkPolyData
    # tube_filter: vtkTubeFilter
    radius: float


class vtkTimerCallbackActors:
    def __init__(self, data: List[AnimationActorData], iren):
        self.timer_count = 0
        self.steps = data[0].points.shape[0]
        self.data = data
        self.iren = iren
        self.timerId = None

    def execute(self, obj, event) -> None:
        step = 0
        while step < self.steps:
            for aad in self.data:
                p = aad.points[step, :]
                aad.sphere_actor.SetPosition(p[0], p[1], p[2])
                # color_data = vtkTimerCallbackActors.create_color_data(aad, step, 10)
                # aad.polydata.GetPointData().AddArray(color_data)
                # aad.tube_filter.Set
            iren = obj
            iren.GetRenderWindow().Render()
            # time.sleep(0.1)
            self.timer_count += 1
            step += 1
        print("end")
        if self.timerId:
            iren.DestroyTimer(self.timerId)

    # @staticmethod
    # def create_color_data(aad:AnimationActorData, step:int, max_len:int)->vtkDoubleArray:
    #     color_data = vtkDoubleArray()
    #     color_data.SetName("color_data")
    #     count_points = aad.points.shape[0]
    #     cdists = np.cumsum(aad.dists)
    #     min_dist = 1e-4
    #     for i in range(count_points):
    #         d = step - i
    #         if d > 0 and d <= max_len:
    #             v = cdists[i-1]/cdists[-1]
    #             color_data.InsertNextValue(max(v,min_dist))
    #         else:
    #             color_data.InsertNextValue(0)
    #     return color_data


class Visualizer:
    enumerate

    def critical_u(H):
        return H * np.exp(H + 1)

    def add_vert(x, color=None, ymax=310):
        plt.plot([x, x], [0, ymax], linewidth=3, color=color)

    def draw_hist(igraph, mrange=(1, 240), rwidth=1, bins=80, xticks=None):
        degree = np.array([d[1] for d in igraph.degree()], dtype=int)

        hist = np.zeros(shape=(degree.max() + 1,), dtype=int)
        for d in degree:
            hist[d] += 1

        plt.hist(
            degree,
            bins=bins,
            histtype='bar',
            range=mrange,
            rwidth=rwidth,
            color='#50ba81',
        )  # 5081ba
        plt.xticks(xticks, fontsize=18)
        plt.yticks(None, fontsize=18)
        return degree, hist

    def draw_nxvtk(
        G,
        node_pos_corr,
        size_node=0.25,
        size_edge=0.02,
        save_pos_path='',
        scale="full_by_1",
        **kwargs,
    ):
        """
        Draw networkx graph in 3d with nodes at node_pos.

        See layout.py for functions that compute node positions.

        node_pos is a dictionary keyed by vertex with a three-tuple
        of x-y positions as the value.

        The node color is plum.
        The edge color is banana.

        All the nodes are the same size.

        @todo to enumerate
        Scale: full_by_1, one_ax_by_1, no

        """

        # Now create the RenderWindow, Renderer and Interactor
        ren = vtkRenderer()

        positions = None
        corr = None
        if type(dict()) == type(node_pos_corr):
            nums = np.array(list(node_pos_corr.keys()), dtype=int)
            positions = np.zeros(shape=(nums.shape[0], 3), dtype=float)
            corr = np.zeros(shape=(nums.max() + 1,), dtype=int)
            i = 0
            for k in node_pos_corr.keys():
                positions[i, :] = node_pos_corr[k]
                corr[int(k)] = i
                i = i + 1
        else:
            positions = node_pos_corr[0]
            corr = node_pos_corr[1]

        data_for_vis = (G, positions, corr)

        focal_pos, camera_pos = Visualizer.draw_graph(
            ren,
            data_for_vis,
            size_node,
            size_edge,
            save_pos_path,
            scale,
            **kwargs,
        )

        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)

        style = vtkInteractorStyleTrackballCamera()
        # style = vtkInteractorStyleFlight()
        # style = vtkInteractorStyleTrackballActor()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        # Add the actors

        camera = ren.GetActiveCamera()
        camera.SetFocalPoint(focal_pos[0], focal_pos[1], focal_pos[2])
        camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])
        # renWin.SetSize(640, 640)

        renWin.Render()
        renWin.Render()
        iren.Initialize()

        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtkTimerCallbackCamera(5000, [], [camera], iren)
            iren.AddObserver('TimerEvent', cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        renWin.Render()
        # renWin.FullScreenOn()
        renWin.SetSize(1900, 1080)
        iren.Start()

    def split_view(
        data1,
        data2,
        size_node=0.25,
        size_edge=0.03,
        save_pos_path='',
        scale="full_by_1",
        **kwargs,
    ):
        xmins = [0, 0.5]
        xmaxs = [0.5, 1]
        ymins = [0] * 2
        ymaxs = [1] * 2

        rw = vtkRenderWindow()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(rw)
        cameras = []
        for i, data in enumerate([data1, data2]):
            ren = vtkRenderer()

            camera = ren.GetActiveCamera()
            camera.SetFocalPoint(0, 0, 0)
            camera.SetPosition(140, 140, 140)
            cameras.append(camera)

            rw.AddRenderer(ren)
            ren.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
            Visualizer.draw_graph(
                ren, data, size_node, size_edge, save_pos_path, scale, **kwargs
            )

        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtkTimerCallbackCamera(5000, [], cameras, iren)
            iren.AddObserver('TimerEvent', cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        rw.SetSize(1900, 1080)
        rw.Render()
        iren.Start()

    def draw_graph(
        ren,
        graph_pos_corr,
        size_node=0.25,
        size_edge=0.02,
        save_pos_path='',
        scale="full_by_1",
        **kwargs,
    ):
        mrange = 1e2
        i = 0

        ndcolors = None
        tdcolors = None
        if len(graph_pos_corr) == 3:
            graph, positions, corr = graph_pos_corr
            if 'colors_data' in kwargs and 'scales_data' in kwargs:
                colors_data = kwargs['colors_data']
                scales_data = kwargs['scales_data']
                ndcolors = [n[1]["color_id"] for n in graph.nodes(data=True)]
                nscales = [
                    scales_data[n[1]["scale_id"]]
                    for n in graph.nodes(data=True)
                ]

        else:
            graph, positions, corr, ndcolors, tdcolors = graph_pos_corr

        a_min = positions.min(axis=0)
        a_max = positions.max(axis=0)
        diff = a_max - a_min
        dmax = diff.max()

        if scale == "full_by_1" or scale == "one_ax_by_1":
            if scale == "full_by_1":
                for i in range(3):
                    positions[:, i] = (
                        positions[:, i] - a_min[i]
                    ) * 2 * mrange / diff[i] - mrange
            elif scale == "one_ax_by_1":
                for i in range(3):
                    d = (positions[:, i] - a_min[i]) / dmax
                    positions[:, i] = ((d - d.max() / 2)) * mrange

        print(f"min = {positions.min(axis=0)}, max = {positions.max(axis=0)}")

        if len(save_pos_path) != 0:
            with open(save_pos_path[0], 'wb') as f:
                np.save(f, positions)
            with open(save_pos_path[1], 'wb') as f:
                np.save(f, corr)

        # set node positions
        colors = vtkNamedColors()
        nodePoints = vtkPoints()
        color_transfer = None
        if colors_data is not None:
            color_transfer = vtkColorTransferFunction()
            for cd, color in colors_data.items():
                color_transfer.AddRGBPoint(cd, color[0], color[1], color[2])

        i = 0
        count_nodes = positions.shape[0]
        pore_data = vtkDoubleArray()
        pore_data.SetNumberOfValues(count_nodes)
        pore_data.SetName("color_data")

        scales = vtkDoubleArray()
        scales.SetNumberOfValues(count_nodes)
        scales.SetName("scales")
        for x, y, z in positions:
            nodePoints.InsertPoint(i, x, y, z)
            if ndcolors is not None:
                pore_data.SetValue(i, ndcolors[i])
            if nscales is not None:
                scales.SetValue(i, nscales[i])
            i = i + 1

        # Create a polydata to be glyphed.
        inputData = vtkPolyData()
        inputData.SetPoints(nodePoints)
        inputData.GetPointData().AddArray(pore_data)
        inputData.GetPointData().AddArray(scales)
        inputData.GetPointData().SetActiveScalars(scales.GetName())

        # Use sphere as glyph source.
        balls = vtkSphereSource()
        balls.SetRadius(size_node)
        balls.SetPhiResolution(20)
        balls.SetThetaResolution(20)

        glyphPoints = vtkGlyph3D()
        glyphPoints.SetInputData(inputData)
        glyphPoints.SetScaleModeToScaleByScalar()
        glyphPoints.SetSourceConnection(balls.GetOutputPort())

        glyphMapper = vtkPolyDataMapper()
        glyphMapper.SetInputConnection(glyphPoints.GetOutputPort())
        glyphMapper.SetScalarModeToUsePointFieldData()
        glyphMapper.SelectColorArray(pore_data.GetName())
        glyphMapper.SetLookupTable(color_transfer)
        glyphMapper.Update()

        glyph = vtkActor()
        glyph.SetMapper(glyphMapper)
        glyph.GetProperty().SetDiffuseColor(1.0, 0.0, 0.0)
        glyph.GetProperty().SetSpecular(0.3)
        glyph.GetProperty().SetSpecularPower(30)

        # Generate the polyline for the spline.
        points = vtkPoints()
        edgeData = vtkPolyData()

        # Edges

        lines = vtkCellArray()
        i = 0
        for u, v in graph.edges():
            # The edge e can be a 2-tuple (Graph) or a 3-tuple (Xgraph)
            lines.InsertNextCell(2)
            for n in (u, v):
                ni = corr[int(n)]
                (x, y, z) = positions[ni, :]
                points.InsertPoint(i, x, y, z)
                lines.InsertCellPoint(i)
                i = i + 1

        edgeData.SetPoints(points)
        edgeData.SetLines(lines)

        # Add thickness to the resulting line.
        Tubes = vtkTubeFilter()
        Tubes.SetNumberOfSides(16)
        Tubes.SetInputData(edgeData)
        Tubes.SetRadius(size_edge)
        #
        profileMapper = vtkPolyDataMapper()
        profileMapper.SetInputConnection(Tubes.GetOutputPort())

        #
        profile = vtkActor()
        profile.SetMapper(profileMapper)
        profile.GetProperty().SetDiffuseColor(0.0, 0.0, 0.0)
        profile.GetProperty().SetSpecular(0.3)
        profile.GetProperty().SetSpecularPower(30)

        ren.AddActor(glyph)
        ren.AddActor(profile)
        ren.SetBackground(colors.GetColor3d("White"))

        mid = positions.mean(axis=0)
        a_min = positions.min(axis=0)
        a_max = positions.max(axis=0)
        diff = a_max - a_min
        return mid, mid + 2 * diff

    def showGraph(
        G,
        size_node=0.25,
        size_edge=0.02,
        layout='kamada',
        save_pos_path='',
        **kwargs,
    ):
        graph = nx.Graph(G)

        # print(f"sqrt={np.sqrt(len(graph.nodes()))}")
        edges = [(i, j, 1) for i, j in graph.edges()]
        graph.add_weighted_edges_from(edges)
        if layout == 'kamada':
            layout = nx.kamada_kawai_layout(graph, dim=3)
        elif layout == 'spring':
            layout = nx.spring_layout(graph, dim=3)
        elif layout == 'spectral':
            layout = nx.spectral_layout(graph, dim=3)
        Visualizer.draw_nxvtk(
            graph,
            layout,
            size_node,
            size_edge,
            save_pos_path=save_pos_path,
            **kwargs,
        )

    @staticmethod
    def create_img_data(
        img: npt.NDArray[np.int8], bbox: Optional[BoundingBox]
    ) -> vtk.vtkImageData:
        image_data = vtk.vtkImageData()
        size = img.shape
        image_data.SetDimensions(size[0], size[1], size[2])
        image_data.AllocateScalars(vtk.VTK_INT, 1)
        if bbox is not None:
            bbs = bbox.size()
            t = [bs / iis for iis, bs in zip(size, bbs)]
            image_data.SetSpacing(*t)
        else:
            image_data.SetSpacing(1, 1, 1)
        for ix in range(size[0]):
            for iy in range(size[1]):
                for iz in range(size[2]):
                    image_data.SetScalarComponentFromDouble(
                        ix, iy, iz, 0, img[ix, iy, iz]
                    )
        return image_data

    @staticmethod
    def create_volume_img(image_data) -> vtk.vtkVolume:
        composite_opacity = vtk.vtkPiecewiseFunction()
        composite_opacity.AddPoint(0, 0.3)
        composite_opacity.AddPoint(1, 0.99)

        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0, 0.26851, 0.009605, 0.335427)
        color_transfer_function.AddRGBPoint(1, 0.993248, 0.906157, 0.143936)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(composite_opacity)
        volume_property.ShadeOff()
        volume_property.SetInterpolationTypeToLinear()

        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        return volume

    @staticmethod
    def create_actor_img(image_data) -> vtkActor:
        marchingcube = vtk.vtkDiscreteFlyingEdges3D()
        marchingcube.SetInputData(image_data)
        marchingcube.ComputeNormalsOn()
        marchingcube.ComputeScalarsOn()
        marchingcube.SetNumberOfContours(1)
        marchingcube.SetValue(0, 1)

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(1)
        lut.SetTableRange(0, 1)
        lut.SetScaleToLinear()
        lut.Build()
        # lut.SetTableValue(0, 0,0,0,0)
        lut.SetTableValue(1, 0, 0, 1, 1)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(marchingcube.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, 2)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor

    @staticmethod
    def add_img_actor(
        ren: vtkRenderer,
        img: npt.NDArray[np.int8],
        volume_mode: bool,
        bbox: Optional[BoundingBox],
    ) -> None:
        image_data = Visualizer.create_img_data(img, bbox)

        mactor = None
        if volume_mode:
            mactor = Visualizer.create_volume_img(image_data)
            ren.AddVolume(mactor)
        else:
            mactor = Visualizer.create_actor_img(image_data)
            ren.AddActor(mactor)

        if bbox is not None:
            pcenter = bbox.center()
            mactor.SetPosition(*pcenter)

    @staticmethod
    def draw_img(
        img: npt.NDArray[np.int8],
        volume_mode: bool,
        bbox: BoundingBox,
        **kwargs,
    ) -> None:
        ren = vtkRenderer()

        Visualizer.add_img_actor(ren, img, volume_mode, bbox)

        colors = vtkNamedColors()
        ren.SetBackground(colors.GetColor3d("White"))

        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)

        style = vtkInteractorStyleTrackballCamera()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        # Add the actors
        center = bbox.center()
        cm_pos = center + bbox.size()

        camera = ren.GetActiveCamera()
        size = img.shape
        camera.SetFocalPoint(*center)
        camera.SetPosition(*cm_pos)
        # renWin.SetSize(640, 640)

        renWin.Render()
        iren.Initialize()

        if 'animation' in kwargs:
            # Sign up to receive TimerEvent
            cb = vtk.vtkTimerCallbackCamera(5000, [], [camera], iren)
            iren.AddObserver('TimerEvent', cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(500)

        renWin.Render()
        # renWin.FullScreenOn()
        renWin.SetSize(1900, 1080)
        iren.Start()

    @staticmethod
    def create_trajectory_actor(
        trj: Trajectory, periodic: bool, color_type: str = 'dist'
    ) -> vtkActor:
        points = trj.points_without_periodic if not periodic else trj.points
        if color_type == 'dist':
            colors = np.cumsum(trj.dists())
            colors = np.append(0, colors)
            colors /= colors[-1]
        elif color_type == 'clusters':
            assert trj.traps is not None
            clusters = trj.traps

            # colors = ndimage.binary_erosion(clusters).astype(clusters.dtype)
            colors = measure.label(clusters, connectivity=1).astype(np.float32)
            # print(colors.max())
            colors /= colors.max()

        return Visualizer.create_polyline_actor(
            points, colors, trj.atom_size * 0.5
        )[0]

    @staticmethod
    def draw_trajectoryes(
        trjs: List[Trajectory],
        color_type='dist',
        periodic: bool = False,
        plot_box: bool = True,
        window_name: str = 'Trajectory',
    ) -> None:
        renderer = vtkRenderer()
        for trj in trjs:
            actor = Visualizer.create_trajectory_actor(
                trj, periodic, color_type
            )
            renderer.AddActor(actor)

        if plot_box:
            outfit_actor = Visualizer.create_box_actor(trjs[0].box)
            renderer.AddActor(outfit_actor)

        colors = vtkNamedColors()

        renderer.SetBackground(colors.GetColor3d("White"))
        renderer.ResetCamera()
        renderer.GetActiveCamera().Azimuth(90)

        renWin = vtkRenderWindow()
        renWin.AddRenderer(renderer)
        renWin.SetSize(640, 512)
        renWin.SetWindowName(window_name)
        renWin.Render()

        style = vtkInteractorStyleTrackballCamera()

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        i = len(interactors)
        interactors.append(iren)
        running.append(True)
        kpis.append(KeyPressInteractorStyle(parent=iren))

        interactors[i].SetInteractorStyle(kpis[i])
        kpis[i].status = running[i]

    @staticmethod
    def show() -> None:
        if len(interactors) == 0:
            return
        interactors[0].Initialize()
        while all(x is True for x in running):
            for i in range(len(kpis)):
                running[i] = kpis[i].status
                if running[i]:
                    interactors[i].ProcessEvents()
                    interactors[i].Render()
                else:
                    interactors[i].TerminateApp()
                    print('Window', i, 'has stopped running.')

    @staticmethod
    def create_box_actor(box: BoundingBox) -> vtkActor:
        lineSource = vtkLineSource()
        lineSource.SetPoint1(box.min())
        lineSource.SetPoint2(box.max())

        colors = vtkNamedColors()
        outline = vtkOutlineFilter()
        outline.SetInputConnection(lineSource.GetOutputPort())
        outlineMapper = vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtkActor()
        outlineActor.SetMapper(outlineMapper)
        outlineActor.GetProperty().SetColor(colors.GetColor3d('Brown'))
        return outlineActor

    @staticmethod
    def create_sphere_actor(
        pos: npt.NDArray[np.float32], radius: float
    ) -> vtkActor:
        colors = vtkNamedColors()
        sphereSource = vtkSphereSource()
        sphereSource.SetCenter(0.0, 0.0, 0.0)
        sphereSource.SetRadius(radius)
        sphereSource.SetPhiResolution(30)
        sphereSource.SetThetaResolution(30)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())
        actor = vtkActor()
        actor.GetProperty().SetColor(colors.GetColor3d("Gray"))
        actor.GetProperty().SetSpecular(0.6)
        actor.GetProperty().SetSpecularPower(30)
        actor.SetMapper(mapper)
        return actor

    @staticmethod
    def create_polyline_actor(
        points: npt.NDArray[np.float32],
        colors: npt.NDArray[np.float32],
        radius: float,
    ) -> vtkActor:
        count_points = points.shape[0]

        ctf = vtkColorTransferFunction()
        ctf.AddRGBPoint(0, 0, 0, 0.0)
        ctf.AddRGBPoint(1e-6, 0, 0, 1.0)
        ctf.AddRGBPoint(0.25, 0, 1.0, 1)
        ctf.AddRGBPoint(0.5, 0, 1, 0)
        ctf.AddRGBPoint(0.75, 1, 1, 0)
        ctf.AddRGBPoint(1.0, 1, 0, 0)

        color_data = vtkDoubleArray()
        color_data.SetName("saturation")

        assert ~np.any(colors > 1)
        assert ~np.any(colors < 0)

        vpoints = vtkPoints()
        lines = vtkCellArray()

        active_throat = 0
        for i in range(1, count_points):
            vpoints.InsertNextPoint(
                points[i - 1, 0], points[i - 1, 1], points[i - 1, 2]
            )
            vpoints.InsertNextPoint(points[i, 0], points[i, 1], points[i, 2])
            line = vtkLine()
            line.GetPointIds().SetId(0, 2 * active_throat)
            line.GetPointIds().SetId(1, 2 * active_throat + 1)
            active_throat += 1

            lines.InsertNextCell(line)
            if len(colors) == count_points:
                c1, c2 = colors[i - 1], colors[i]
            else:
                c1, c2 = colors[i - 1], colors[i - 1]

            color_data.InsertNextValue(c1)
            color_data.InsertNextValue(c2)

        poly_data = vtkPolyData()
        poly_data.SetPoints(vpoints)
        poly_data.SetLines(lines)
        poly_data.GetPointData().AddArray(color_data)

        tube_filter = vtkTubeFilter()
        tube_filter.SetNumberOfSides(16)
        tube_filter.SetInputData(poly_data)
        tube_filter.SetRadius(radius * 0.01)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(color_data.GetName())
        mapper.SetLookupTable(ctf)
        mapper.Update()

        actor = vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().SetDiffuse(0.7)
        # actor.GetProperty().SetSpecular(0.4)
        # actor.GetProperty().SetSpecularPower(1)
        actor.GetProperty().SetDiffuse(0)
        actor.GetProperty().SetSpecular(0)
        actor.GetProperty().SetAmbient(1)
        # actor.GetProperty().SetSpecularPower(0)
        actor.GetProperty().BackfaceCullingOn()
        return actor, poly_data, tube_filter

    @staticmethod
    def draw_trajectory_points(trj: Trajectory) -> None:
        tp = trj.points_without_periodic
        pcount = tp.shape[0]

        points = vtkPoints()
        points.Resize(pcount)
        points.SetNumberOfPoints(pcount)

        pdata = vtkDoubleArray()
        pdata.SetName("clusters")
        pdata.SetNumberOfValues(pcount)

        if trj.traps is not None:
            count_clusters = trj.traps.max() + 1

        for i in range(pcount):
            points.SetPoint(i, tp[i, 0], tp[i, 1], tp[i, 2])
            if trj.traps is not None:
                pdata.SetValue(i, float(trj.traps[i]) / trj.traps.max())
            else:
                pdata.SetValue(i, 1.0)
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(pdata)

        # const double pore_scale = vis_set->poreScaleRadius();
        sphere_source = vtkSphereSource()
        sphere_source.SetRadius(trj.atom_size * 0.1)
        glyph = vtkGlyph3D()
        # pore_glyph.SetScaleFactor(pore_scale)
        glyph.SetSourceConnection(sphere_source.GetOutputPort())
        glyph.SetInputData(polydata)

        ctf = vtkColorTransferFunction()
        if trj.traps is not None:
            count_clusters = trj.traps.max() + 1
            for i in range(count_clusters):
                ctf.AddRGBPoint(
                    float(i) / trj.traps.max(),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                )
        else:
            ctf.AddRGBPoint(0, 0, 0, 1.0)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(pdata.GetName())
        mapper.SetLookupTable(ctf)
        mapper.Update()

        actor = vtkActor()
        actor.SetMapper(mapper)

        renderer = vtkRenderer()
        renderer.AddActor(actor)

        trj_actor = Visualizer.create_trajectory_actor(trj, False)
        renderer.AddActor(trj_actor)

        colors = vtkNamedColors()

        renderer.SetBackground(colors.GetColor3d("White"))
        renderer.ResetCamera()
        renderer.GetActiveCamera().Azimuth(90)

        renWin = vtkRenderWindow()
        renWin.AddRenderer(renderer)
        renWin.SetSize(1900, 1060)
        renWin.SetWindowName('Trajectory')
        renWin.Render()

        style = vtkInteractorStyleTrackballCamera()

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)
        iren.Initialize()
        iren.Start()

    @staticmethod
    def draw_img_trj(
        img: npt.NDArray[np.int8],
        bbox: BoundingBox,
        trj: Trajectory,
        volume_mode,
    ) -> None:
        ren = vtkRenderer()

        Visualizer.add_img_actor(ren, img, volume_mode, bbox)

        actor = Visualizer.create_trajectory_actor(trj, False)
        ren.AddActor(actor)

        colors = vtkNamedColors()
        ren.SetBackground(colors.GetColor3d("White"))

        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)

        style = vtkInteractorStyleTrackballCamera()
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)

        # Add the actors
        size = img.shape
        fpos = size[0] / 2, size[1] / 2, size[2] / 2
        cpos = size[0] * 2, size[1] * 2, size[2] * 2
        if bbox is not None:
            fpos = bbox.center()
            cpos = bbox.center() + np.array([*(bbox.size())])

        camera = ren.GetActiveCamera()
        camera.SetFocalPoint(*fpos)
        camera.SetPosition(*cpos)
        # renWin.SetSize(640, 640)

        iren.Initialize()
        renWin.Render()
        renWin.SetSize(1900, 1080)
        iren.Start()

    @staticmethod
    def animate_trajectoryes(
        trjs: List[Trajectory], periodic: bool = False, plot_box: bool = True
    ) -> None:
        def discr(
            points: npt.NDArray[np.float32], count: int
        ) -> npt.NDArray[np.float32]:
            npoints = np.zeros(
                shape=((count + 1) * (points.shape[0] - 1) + 1, 3),
                dtype=np.float32,
            )
            for i in range(points.shape[0] - 1):
                d = points[i + 1] - points[i]
                for j in range(count + 1):
                    npoints[j + i * (count + 1)] = points[i] + j * d / (
                        count + 1
                    )
            npoints[-1] = points[-1]
            return npoints

        renderer = vtkRenderer()
        data = []
        for trj in trjs:
            radius = trj.atom_size * 0.25
            p = trj.points if periodic else trj.points_without_periodic
            sphere_actor = Visualizer.create_sphere_actor(
                p[0, :], trj.atom_size
            )

            renderer.AddActor(sphere_actor)
            npoints = discr(p, 5)
            data.append(AnimationActorData(sphere_actor, npoints, radius))
        if plot_box:
            outfit_actor = Visualizer.create_box_actor(trjs[0].box)
            renderer.AddActor(outfit_actor)

        colors = vtkNamedColors()

        renderer.SetBackground(colors.GetColor3d("White"))
        renderer.ResetCamera()
        renderer.GetActiveCamera().Azimuth(90)

        renWin = vtkRenderWindow()
        renWin.AddRenderer(renderer)
        renWin.SetSize(1900, 1060)
        renWin.SetWindowName('Trajectory')

        style = vtkInteractorStyleTrackballCamera()

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)
        iren.Initialize()

        cb = vtkTimerCallbackActors(data, iren)
        iren.AddObserver('TimerEvent', cb.execute)
        cb.timerId = iren.CreateRepeatingTimer(2000)

        renWin.Render()
        iren.Start()
