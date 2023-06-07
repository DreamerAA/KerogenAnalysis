import math as m

import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import vtk
import vtkmodules.vtkRenderingOpenGL2
import xarray as xr
from matplotlib import pylab
from matplotlib.ticker import MaxNLocator
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints, vtkIdList, vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkTubeFilter, vtkPolyDataNormals
from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkLineSource
from vtkmodules.vtkFiltersModeling import vtkRotationalExtrusionFilter, vtkOutlineFilter
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleFlight,
    vtkInteractorStyleTrackballActor,
    vtkInteractorStyleTrackballCamera,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
from trajectory import Trajectory
from boundingbox import BoundingBox
from typing import List
from dataclasses import dataclass
import time
import random


def Rx(theta):
    return np.matrix(
        [
            [1, 0, 0],
            [0, m.cos(theta), -m.sin(theta)],
            [0, m.sin(theta), m.cos(theta)],
        ]
    )


def Ry(theta):
    return np.matrix(
        [
            [m.cos(theta), 0, m.sin(theta)],
            [0, 1, 0],
            [-m.sin(theta), 0, m.cos(theta)],
        ]
    )


def Rz(theta):
    return np.matrix(
        [
            [m.cos(theta), -m.sin(theta), 0],
            [m.sin(theta), m.cos(theta), 0],
            [0, 0, 1],
        ]
    )


class vtkTimerCallbackCamera:
    def __init__(self, steps, actors, cameras, iren):
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

    def calcXYZ(self):
        a = self.angle * np.pi / 180
        self.Rxyz = Ry(a) * Rx(a)
        return (self.cur_pos * self.Rxyz).T.A.squeeze()

    def execute(self, obj, event):
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
    radius:float

class vtkTimerCallbackActors():
    def __init__(self, data:List[AnimationActorData], iren):
        self.timer_count = 0
        self.steps = data[0].points.shape[0]
        self.data = data
        self.iren = iren
        self.timerId = None 

    def execute(self, obj, event)->None:
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
        )  ##5081ba
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
        count_edges = len(graph.edges())
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

    def z(c, H, u, p):
        return 2 * (c**H) * (u * (p**c) + 1) / (u * p + 1)

    def z_p():
        c, H = 4, 4
        du = np.arange(3, 7)
        u = 10**du
        p = np.arange(0.0, 0.4, 0.05)
        for u_el in u:
            plt.plot(p, z(c, H, u_el, p))

        plt.plot()

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

    # 'viridis', 'RdBu', 'Spectral', 'bwr', 'seismic'

    def showRegularResult(
        data_path, xticks=None, yticks=None, field="lnz", log_callback=None
    ):
        df = xr.load_dataset(data_path)
        res = df[field]

        if log_callback != None:
            log_callback(res)

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        r = res.plot(cmap='seismic')

        lbls = list(df.dims.keys())

        plt.xlabel(lbls[0], fontsize=20)
        plt.ylabel(lbls[1], fontsize=20)
        plt.xticks(xticks, fontsize=18)
        plt.yticks(yticks, fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

    def add_critical_u(h, max_p=1):
        u = np.log10(Visualizer.critical_u(h))
        print(u)
        plt.plot([0, max_p], [u, u], color='darkgreen', linewidth=3.0)

    def draw_img(img: npt.NDArray[np.int8], volume_mode, **kwargs):
        image_data = vtk.vtkImageData()
        size = img.shape
        image_data.SetDimensions(size[0], size[1], size[2])
        image_data.AllocateScalars(vtk.VTK_INT, 1)
        image_data.SetSpacing(1, 1, 1)
        for ix in range(size[0]):
            for iy in range(size[1]):
                for iz in range(size[2]):
                    image_data.SetScalarComponentFromDouble(
                        ix, iy, iz, 0, img[ix, iy, iz]
                    )

        colors = vtkNamedColors()
        ren = vtkRenderer()
        #

        ren.SetBackground(colors.GetColor3d("White"))

        if volume_mode:
            composite_opacity = vtk.vtkPiecewiseFunction()
            composite_opacity.AddPoint(0, 0.01)
            composite_opacity.AddPoint(1, 1)

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

            ren.AddVolume(volume)
        else:
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

            ren.AddActor(actor)

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
        camera.SetFocalPoint(size[0] / 2, size[1] / 2, size[2] / 2)
        camera.SetPosition(size[0] * 2, size[1] * 2, size[2] * 2)
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
    def create_trajectory_actor(trj:Trajectory, periodic:bool)->vtkActor:
        points = trj.points_without_periodic() if not periodic else trj.points
        return Visualizer.create_polyline_actor(points, trj.dists(), trj.atom_size*0.5)[0]
        
    @staticmethod
    def draw_trajectoryes(trjs:List[Trajectory], periodic:bool = False, plot_box:bool=True)->None:

        renderer = vtkRenderer()
        for trj in trjs:
            actor = Visualizer.create_trajectory_actor(trj, periodic)
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
        renWin.SetWindowName('Trajectory')
        renWin.Render()


        style = vtkInteractorStyleTrackballCamera()

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.SetInteractorStyle(style)
        iren.Start()

    @staticmethod
    def create_box_actor(box:BoundingBox)->vtkActor:
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
    def create_sphere_actor(pos: npt.NDArray[np.float32], radius:float)->vtkActor:
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
    def create_polyline_actor(points: npt.NDArray[np.float32], dists: npt.NDArray[np.float32], radius:float)->vtkActor:
        count_points = points.shape[0]

        # ctf = vtkColorTransferFunction()
        # ctf.AddRGBPoint(-1, 0, 0, 0)
        # ctf.AddRGBPoint(0, 0, 0, 1.)
        # ctf.AddRGBPoint(0.25, 0, 1., 1)
        # ctf.AddRGBPoint(0.5, 0, 1, 0)
        # ctf.AddRGBPoint(0.75, 1, 1, 0)
        # ctf.AddRGBPoint(1., 1, 0, 0)

        lut = vtkLookupTable()
        lut.SetNumberOfColors(9)
        lut.SetTableRange(-1, 1)
        lut.SetScaleToLinear()
        lut.Build()

        mid_dist = 1e-4
        lut.SetTableValue(0, 0, 0, 0, 0. )
        lut.SetTableValue(1, 0, 0, 0, 0. )
        lut.SetTableValue(2, 0, 0, 0, 0. )
        lut.SetTableValue(3, 0, 0, 0, 0. )
        lut.SetTableValue(4, 0, 0, 1., 1.)
        lut.SetTableValue(5, 0, 1., 1, 1.)
        lut.SetTableValue(6, 0, 1, 0, 1.)
        lut.SetTableValue(7, 1, 1, 0, 1.)
        lut.SetTableValue(8, 1, 0, 0, 1.)


        color_data = vtkDoubleArray()
        color_data.SetName("saturation")
        min_dist = 0.000
        cdists = np.cumsum(dists) + min_dist

        vpoints = vtkPoints()
        lines = vtkCellArray()
        # radiuses = vtkDoubleArray()
        # radiuses.SetName("radiuses")


        lines.InsertNextCell(count_points)
        for i in range(count_points):
            vpoints.InsertPoint(i,points[i,0],points[i,1],points[i,2])
            lines.InsertCellPoint(i)
            
            if i != 0:
                v = cdists[i-1]/cdists[-1]
                color_data.InsertNextValue(max(v,mid_dist))
                # radiuses.InsertNextTuple1(radius*v/min_dist)
                # radiuses.InsertNextTuple1(radius*v)
                
            else:
                # radiuses.InsertNextTuple1(min_dist)
                color_data.InsertNextValue(mid_dist)
            
        poly_data = vtkPolyData()
        poly_data.SetPoints(vpoints)
        poly_data.SetLines(lines)
        poly_data.GetPointData().AddArray(color_data)
        # poly_data.GetPointData().AddArray(radiuses)
        # poly_data.GetPointData().SetActiveScalars(radiuses.GetName())

        tube_filter = vtkTubeFilter()
        tube_filter.SetNumberOfSides(16)
        tube_filter.SetInputData(poly_data)
        tube_filter.SetRadius(radius)
        # tube_filter.SetVaryRadiusToVaryRadiusByScalar()
        # tube_filter.SetRadiusFactor(radius)


        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(color_data.GetName())
        mapper.SetLookupTable(lut)
        

        
        actor = vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(colors.GetColor3d("PowderBlue"))
        actor.GetProperty().SetDiffuse(0.7)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(20)
        actor.GetProperty().BackfaceCullingOn()
        return actor, poly_data, tube_filter

    @staticmethod
    def draw_trajectory_points(trj:Trajectory):
        pcount = trj.shape[0]
        points = vtkPoints()
        points.Resize(pcount)
        points.SetNumberOfPoints(pcount)
        pdata = vtkDoubleArray()
        tp = trj.points
        for i in range(pcount):
            points.SetValue(i, tp[i,0], tp[i,1], tp[i,2])
            if trj.clusters:
                pdata.SetValue(i, trj.clusters[i])
            else:
                pdata.SetValue(i, 1.)
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(pdata)

        # const double pore_scale = vis_set->poreScaleRadius();
        sphere_source = vtkSphereSource()
        sphere_source.SetRadius(trj.atom_size*0.01)
        glyph = vtkGlyph3D()
        # pore_glyph.SetScaleFactor(pore_scale)
        glyph.SetSourceConnection(sphere_source.GetOutputPort())
        glyph.SetInputData(polydata)

        ctf = vtkColorTransferFunction()
        if trj.clusters:
            for i in range(trj.clusters.max() +1):
                ctf.AddRGBPoint(i, random.uniform(0,1), random.uniform(0,1), random.uniform(0,1));    
        else:
            ctf.AddRGBPoint(0, 0, 0, 1.); 
    
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
    def animate_trajectoryes(trjs:List[Trajectory], periodic:bool = False, plot_box:bool=True)->None:

        def discr(points: npt.NDArray[np.float32], count:int)->npt.NDArray[np.float32]:
            npoints = np.zeros(shape=((count + 1)*(points.shape[0] - 1) + 1,3),dtype=np.float32)
            for i in range(points.shape[0] - 1):
                d = points[i + 1] - points[i]
                for j in range(count + 1):
                    npoints[j + i *(count+1)] = points[i] + j*d/(count+1)
            npoints[-1] = points[-1]
            return npoints

        renderer = vtkRenderer()
        data = []
        for trj in trjs:
            radius = trj.atom_size*0.25
            p = trj.points if periodic else trj.points_without_periodic()
            sphere_actor = Visualizer.create_sphere_actor(p[0,:], trj.atom_size)
            # polyline_actor, polydata, tube_filter = Visualizer.create_polyline_actor(p, trj.dists(), radius)
            
            renderer.AddActor(sphere_actor)
            # renderer.AddActor(polyline_actor)
            npoints = discr(p,5)
            data.append(AnimationActorData(sphere_actor, 
                                        #    polyline_actor, 
                                           npoints, 
                                        #    Trajectory.extractDists(npoints), 
                                        #    polydata,
                                        #    tube_filter, 
                                           radius))
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