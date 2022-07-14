import numpy as np
import k3d

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