from typing import Callable, List
from functools import partial
from pathlib import Path
import json
import os

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
    if starting_view is None:
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
    path_to_objs: List[Path],
    outpout_dir: Union[str, Path] = 'data/data/ShapeNetCore.v2_nbv',
    num_voxels: int = 32,
    num_starting_views: int = 1,
    sphere_radius: float = np.sqrt(3.1),  # encompasses unit square
    coverage_thresh: float = 0.7, 
    max_iter: int = 10,
    camera_type: str = "kinect",
    camera_intrinsic_mtx: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic] = None,
    debug: bool = False
):
    """Function for generating data.

    First, 3D objects are read from the paths given with `path_to_objs`. Then a ground truth voxel grid
    is created from each of them. Then, the following algorithm is applied:
    ```
    Given: Ground Truth Model G
    1. Create a list of random starting views LS on a sphere around G
    2. For each starting view sv in LS
        1. Init partial model P similar to G but with unseen voxels
        2. Scan G from sv and incorporate results into P
        3. While P is incomplete and total number of scans < max_iter
            1. Find NBV v*  // Algorithm 2
            2. Store the data tuple (P, v*, direction(v*), utility(v*))
            3. Scan G from v*
            4. Incorporate discovered occupied and empty voxels into P
        5. If P is complete, dump the data tuples.

    ```
    """
    nbv_fn = partial(
        next_best_view_exhaustive,
        utility_fn = num_discovered_occupied_voxels,
    )   

    for idx, obj_path in enumerate(path_to_objs):
        synset_id, obj_id = obj_path.parts[-4:-2]
        if debug:
            print(f"\nProcessing {synset_id}/{obj_id}\t[{idx+1}/{len(path_to_objs)}]")
        # setup ground truth grid
        try:
            obj_mesh = o3d.io.read_triangle_mesh(str(obj_path))
            grid_data, voxel_size, min_bound, max_bound = to_occupancy_grid(
                obj_mesh, input_type="mesh", do_normalization=True, num_voxels=num_voxels
            )   
        except:
            print(f"Error loading {obj_path}. Skipping")
            continue
        gt_grid = Grid(voxel_size, num_voxels, Vec3D(min_bound), Vec3D(max_bound))
        gt_grid.data = grid_data
        if debug:
            print("Finished setting the Ground Truth Grid")

        starting_views = random_sphere_point(npoints=num_starting_views, radius=sphere_radius)
        for sview_idx, sview in enumerate(starting_views):
            if debug:
                print(f"Scan session {sview_idx+1}/{num_starting_views} from starting view {sview}")
            obj_dict = dict()
            for scan_idx, scan in enumerate(scan_object(
                gt_grid, 
                nbv_fn,
                starting_view=sview,
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
                        print(f"\tReceived scan no {scan_idx}")
                        for key, val in sample_info.items():
                            if key == "partial_model":
                                num_occupied = (np.array(val) == VoxelType.occupied).sum()
                                print(f"\t\t# of occed : {num_occupied}")
                            else:
                                print(f"\t\t{key} : {val}")
                        print()
                    obj_dict[scan_idx] = sample_info
                else:
                    coverage = (partial_grid_data == VoxelType.occupied).sum() \
                            / (gt_grid.data == VoxelType.occupied).sum()
                    # discard the ones with inadequate coverage
                    if coverage > coverage_thresh:
                        file_dir_path = Path(outpout_dir, synset_id, obj_id, str(sview_idx))
                        if debug:
                            print(f"\tCoverage is {coverage}, writing data to {file_dir_path}")
                        os.makedirs(file_dir_path, exist_ok=True)
                        for scan_idx, scan_dict in obj_dict.items():
                            file_path = Path(file_dir_path / f'{scan_idx}.json')
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(scan_dict, f, ensure_ascii=False, indent=4)
