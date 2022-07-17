from typing import Union

import numpy as np
import k3d
import open3d as o3d

from .data_utils import flip_axes


def visualize_occupancy(occupancy_grid, point_size=1, do_flip_axes=False):
    if not isinstance(occupancy_grid, (list, tuple)):
        occupancy_grid =  [occupancy_grid]
    points_list = list()
    for og in occupancy_grid:
        points = np.concatenate([c[:, np.newaxis] for c in np.where(og)], axis=1)
        points_list.append(points)
    visualize_pointcloud(points_list, point_size, do_flip_axes, name='occupancy_grid')


def visualize_pointcloud(point_clouds, point_size=0.01, do_flip_axes=False, name='point_cloud'):
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]
    plot = k3d.plot(name=name, grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if do_flip_axes:
        point_clouds = [flip_axes(pc) for pc in point_clouds]

    colors = np.random.uniform(low=0., high=1., size=len(point_clouds)) * (2 ** 24)
    for pc, color in zip(point_clouds, colors):
        color = int(color)
        plt_points = k3d.points(positions=pc.astype(np.float32), point_size=point_size, color=color)
        plot += plt_points
    plt_points.shader ='3d' 
    plot.display()


def visualize_camera_path(path: Union[list, np.ndarray], sphere_radius=np.sqrt(3)):
    plt = k3d.plot()

    # create a sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=sphere_radius, resolution=10, create_uv_map=False)
    vertices = np.array(sphere.vertices)
    indices  = np.array(sphere.triangles)
    plt += k3d.lines(
        vertices, indices, indices_type='triangle',
        shader='mesh', width=0.003, color=0x570861, opacity=0.7)
    # royal green: 0x126108

    # plot the camera path
    if isinstance(path, list):
        path = np.vstack(path)
    indices = np.array([[i, i+1] for i in range(len(path) - 1)])
    plt += k3d.lines(
        path, indices, indices_type='segment', 
        shader='mesh', width=0.015, color=0x126108,
        # color_map= k3d.colormaps.matplotlib_color_maps.summer,
        # attribute=np.log(np.arange(len(indices))),
        opacity=0.7
    )

    # plot camera points
    point_sizes = np.ones(len(path)) * 0.15
    point_sizes[0] = 0.25
    plt += k3d.points(
        path, 
        point_sizes=point_sizes,
        attribute=np.log(np.arange(len(path))+1),
        color_map= k3d.colormaps.matplotlib_color_maps.summer,
    )

    plt.display()