from typing import Callable
from functools import partial
from pathlib import Path
import json

import open3d as o3d
import numpy as np
import torch.nn as nn

from utils.data_utils import *
from ray_tracing import *


# a full circle is traversed in STEP_FACTOR * 2 steps (see STEP_SIZE)
STEP_FACTOR = 4
PI = np.pi
STEP_SIZE = PI / STEP_FACTOR  # in radians


def move(direction, location):
    """ Calculates the new location after moving from the given location in the
    given direction.

    Movement happens on a sphere. Up, down, left, and right correspond to north,
    south, east, and west. Up/down movements happen along the longitude,
    while left/right movements happen along the latitude. If close to a pole,
    left/right movements are restricted to one-third of the perimeter of the
    current latitude circle to avoid moving full-circle in a single step. 
    Up/down movements can go across a pole. Since there is no direction
    at the poles, movement does not stop at a pole and goes a bit further.

    Parameters
    ----------
    direction : str
        One of 'up', 'down', 'right', 'left'.
    location : np.array
        Current location in cartesian coordinates (x, y, z).

    Returns
    -------
    np.array
        New location after moving, in cartesian coordinates.
    """

    assert direction in ["up", "down", "right", "left"]
    
    r, theta, phi = cartesian2spherical(location)
    if direction == "up":
        theta -= STEP_SIZE
        # check if at the pole, if so move further
        if np.isclose(theta, 0., atol=1e-2):
            theta -= 0.05 * STEP_SIZE
    elif direction == "down":
        theta += STEP_SIZE
        # check if at the pole, if so move a bit further
        if np.isclose(theta, PI, atol=1e-2):
            theta += 0.05 * STEP_SIZE

    # check if moved across a pole and changed hemisphere
    if theta < 0:
        theta = np.abs(theta) % PI
        # shift 180 degrees to change hemisphere
        phi = (phi + PI) % (2 * PI)

    if theta > PI:
        theta = theta - theta % PI
        # shift 180 degrees to change hemisphere
        phi = (phi + PI) % (2 * PI) 
    
    if direction == "left" or direction == "right":
        step_arc_length = STEP_SIZE * r
        latitude_radius = r * np.sin(theta)
        latitude_perimeter = 2 * PI * latitude_radius
        # restrict the step size to one-third
        step_size = min(step_arc_length, latitude_perimeter / 3)
        step_size_radians = step_size / latitude_radius
        if direction == "left":
            phi -= step_size_radians
        else:
            phi += step_size_radians
        phi %= (2 * PI)  # fix the range back to [0, 2PI]
    
    new_location = spherical2cartesian(np.array([r, theta, phi]))
    return new_location


def num_discovered_occupied_voxels(
    view: np.ndarray, 
    partial_grid: Grid, 
    gt_grid: Grid,
    camera_intrinsic: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic]
):
    num_occ_before = (partial_grid.data == VoxelType.occupied).sum()
    rays = generate_rays(view, camera_intrinsic)
    trace_rays(rays, partial_grid, gt_grid, max_t_threh=5)
    num_discovered = (partial_grid.data == VoxelType.occupied).sum() - num_occ_before
    return num_discovered


def next_best_view_nn(
    model: nn.Module,
    current_view: np.ndarray,
    partial_grid: Grid,
    gt_grid: Grid,
    utility_fn: Callable,
    camera_intrinsic: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic]
):
    pass


def next_best_view_exhaustive(
    current_view: np.ndarray,
    partial_grid: Grid,
    gt_grid: Grid,
    utility_fn: Callable,
    camera_intrinsic: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic]
):
    best_nview = None
    best_npartial_grid = None
    best_utiliy = -np.inf
    best_nview_dir = None
    for (direction, nview) in [
        (direction, move(direction, current_view)) \
            for direction in ["up", "down", "right", "left"]
    ]:
        npartial_grid = partial_grid.copy()
        utility = utility_fn(nview, npartial_grid, gt_grid, camera_intrinsic)
        if utility > best_utiliy:
            best_utiliy = utility
            best_nview = nview
            best_npartial_grid = npartial_grid
            best_nview_dir = direction
    return best_nview_dir, best_nview, best_npartial_grid, best_utiliy
    

def scan_object(
    gt_grid: Grid,
    nbv_fn: Callable,
    camera_type: str = None,
    camera_intrinsic_mtx: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic] = None,
    starting_view: np.ndarray = None,
    sphere_radius: float = np.sqrt(3.1), # encompasses unit square
    coverage_thresh: float = 0.7, 
    max_iter: int = 10,
):
    if not starting_view:
        view = random_sphere_point(npoints=1, radius=sphere_radius).flatten()
    else:
        view = starting_view

    if camera_type:
        assert camera_intrinsic_mtx is None
        camera_intrinsic_mtx = init_camera_intrinsic(camera_type)
    else:
        assert camera_intrinsic_mtx is not None

    # initialize the partial grid
    partial_grid = Grid(gt_grid.voxel_size, gt_grid.num_voxels,
                         gt_grid.min_bound, gt_grid.max_bound)
    rays = generate_rays(view, camera_intrinsic_mtx)
    trace_rays(rays, partial_grid, gt_grid, max_t_threh=5)

    num_occupied = (gt_grid.data == VoxelType.occupied).sum()
    iter = 0
    coverage = 0
    while (coverage < coverage_thresh and iter < max_iter):
        best_nview_dir, best_nview, best_npartial_grid, best_utility = nbv_fn(
            current_view=view, partial_grid=partial_grid, gt_grid=gt_grid, camera_intrinsic=camera_intrinsic_mtx)
        yield best_nview_dir, best_nview, partial_grid.data, best_utility

        partial_grid.data = best_npartial_grid.data
        view = best_nview
        coverage = (partial_grid.data == VoxelType.occupied).sum() / num_occupied
        iter += 1
    yield None, None, partial_grid.data, None


def generate_data(
    mesh_object: o3d.geometry.TriangleMesh,
    gt_grid: Grid,  # temporary
    num_voxels: int = 32,
    sphere_radius: float = np.sqrt(3.1),  # encompasses unit square
    coverage_thresh: float = 0.7, 
    max_iter: int = 10,
    camera_type: str = "kinect",
    camera_intrinsic_mtx: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic] = None,
    debug: bool = False
):
    """
    Given an object O:
    1. Create an initial partial model M
        1. Center O at (0, 0, 0)
        2. Create an empty 32x32x32 voxel grid M - partial model, where each voxel is initally labeled as `unseen`
        3. Pick a random view v on the sphere encompassing O
        4. Generate rays from this view
        5. Label each voxel in M via tracing each ray as either empty or occupied
    2. Until the object is fully seen, i.e., no unseen voxels
        1. Find the next views - views to the North, East, West, and South of the current view v
        2. For each next view
            1. Generate rays from the view
            2. Label each voxel in M creating M' via tracing each ray
            3. Utility of this view = (# of unseen voxels in M) - (# of unseen voxels in M')
        3. Pick the view with the best utility v* and dump (M, v*)
        4. Update M, M = M'(created with rays from v*)
    """

    if not gt_grid:
        # setup ground truth grid
        grid_data, voxel_size, min_bound, max_bound = to_occupancy_grid(
            mesh_object, input_type="mesh", do_normalization=True, num_voxels=num_voxels
        )
        gt_grid = Grid(voxel_size, num_voxels, Vec3D(min_bound), Vec3D(max_bound))
        gt_grid.data = grid_data
        if debug:
            print("Finished setting the Ground Truth Grid")
    
    nbv_fn = partial(
        next_best_view_exhaustive,
        utility_fn = num_discovered_occupied_voxels,
    )

    obj_dict = dict()
    for idx, scan in enumerate(scan_object(
        gt_grid, 
        nbv_fn,
        camera_type=camera_type,
        camera_intrinsic_mtx=camera_intrinsic_mtx,
        sphere_radius=sphere_radius, 
        coverage_thresh=coverage_thresh, 
        max_iter=max_iter
    )):
        best_view_dir, best_view, partial_grid_data, best_utility = scan
        if best_view_dir:  # is None if the scanning is over
            sample_info = {
                "next_view_dir"     : best_view_dir,
                "next_view_coords"  : best_view.tolist(),
                "partial_model"     : partial_grid_data.tolist(),
                "next_view_utility" : int(best_utility)
            }
            if debug:
                print(f"Received scan no {idx}")
                for key, val in sample_info.items():
                    if key == "partial_model":
                        num_occupied = (np.array(val) == VoxelType.occupied).sum()
                        print(f"\t# of occed : {num_occupied}")
                    else:
                        print(f"\t{key} : {val}")
                print()
            obj_dict[idx] = sample_info
        else:
            coverage = (partial_grid_data == VoxelType.occupied).sum() \
                    / (gt_grid.data == VoxelType.occupied).sum()
            # discard the ones with inadequate coverage
            if coverage > coverage_thresh:
                file_name = Path('data/data/trial/bun.json')
                if debug:
                    print(f"Coverage is {coverage}, writing data to {file_name}")
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(obj_dict, f, ensure_ascii=False, indent=4)
