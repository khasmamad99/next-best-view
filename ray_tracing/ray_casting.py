import numpy as np
import open3d as o3d

from .classes import *


def generate_rays(
    eye: np.ndarray,
    camera_intrinsic: o3d.camera.PinholeCameraIntrinsic,
    center: np.ndarray = np.array([0, 0, 0]),
    up: np.ndarray = np.array([0, 1, 0]),
) -> Vec3D:
    fx, _ = camera_intrinsic.get_focal_length()
    x0, y0 = camera_intrinsic.get_principal_point()
    width, height = x0 * 2, y0 * 2
    fov = np.arctan2(x0, fx) * 180 / np.pi

    scene = o3d.t.geometry.RaycastingScene()
    rays = scene.create_rays_pinhole(fov, center, eye, up, width, height)
    rays = rays.numpy().reshape(-1, 6)
    rays = Ray(origin=Vec3D(rays[:, :3]), direction=Vec3D(rays[:, 3:]))
    return rays


def intersection_points(rays, grid):
    """Returns the minimum distance along the rays to enter and exit the grid, 
    t_min and t_max. If there is no intersection, t_min and t_max are set to inf.
    """
    t_min = np.ones([len(rays),], dtype=np.float32) * -np.inf
    t_max = np.ones([len(rays),], dtype=np.float32) * np.inf

    t_dir_min = np.empty([len(rays),], dtype=np.float32)
    t_dir_max = np.empty([len(rays),], dtype=np.float32)

    # TO DO: what happens if origin is inside the bounds?
    for origin, dir, min_bound, max_bound in zip(
        rays.origin.data.T, rays.direction.data.T,
        grid.min_bound.data, grid.max_bound.data
    ):
        inv_dir = 1. / dir  # division by 0 results in inf
        # TO DO: think through the implications of the following change
        inv_dir[inv_dir == np.inf] = 0
        
        t_dir_min[inv_dir >= 0] = ((min_bound - origin) * inv_dir)[inv_dir >= 0]
        t_dir_min[inv_dir <  0] = ((max_bound - origin) * inv_dir)[inv_dir <  0]

        t_dir_max[inv_dir >= 0] = ((max_bound - origin) * inv_dir)[inv_dir >= 0]
        t_dir_max[inv_dir <  0] = ((min_bound - origin) * inv_dir)[inv_dir <  0]

        t_min[np.logical_or(t_min > t_dir_max, t_dir_min > t_dir_max)] = np.inf # does not intersect
        mask = t_dir_min > t_min
        t_min[mask] = t_dir_min[mask]
        mask = t_dir_max < t_max
        t_max[mask] = t_dir_max[mask]

    return t_min, t_max


def trace_rays(rays, partial_grid, gt_grid, max_t_threh, min_t_thresh=0):
    """Shoots the given rays towards `gt_grid` and labels `partial_grid`. 
    
    The voxels in `partial_grid` are labeled as `empty` if a ray passes through 
    the corresponding voxel in `gt_grid`. If a ray hits a voxel in `gt_grid`, 
    the corresponding voxel in `partial_grid` is labeled as `occupied`. 
    All the remaining voxels in `partial_grid` remain labeled `unseen`.

    This implementation borrows a lot from 
    https://github.com/cgyurgyik/fast-voxel-traversal-algorithm/blob/master/amanatidesWooAlgorithm.cpp.
    Unlike the above implementation, here ray tracing is done for all rays at once
    in a numpy vectorized fashion.
    """
    t_min, t_max = intersection_points(rays, partial_grid)
    hits = np.logical_and(t_min != -np.inf, t_max != np.inf)
    t_min = np.clip(t_min, a_min=min_t_thresh, a_max=max_t_threh)
    t_max = np.clip(t_max, a_min=min_t_thresh, a_max=max_t_threh)
    ray_start = Vec3D((rays.origin.data + t_min[:, None] * rays.direction.data)[hits])
    ray_end = Vec3D((rays.origin.data + t_max[:, None] * rays.direction.data)[hits])

    def init_params(start, end, direction, min_bound):
        current_index = np.clip(
            np.floor((start - min_bound) / partial_grid.voxel_size), 
            a_min=0, 
            a_max=partial_grid.num_voxels - 1
        ).astype(np.int8)
        end_index = np.clip(
            np.floor((end - min_bound) / partial_grid.voxel_size), 
            a_min=0, 
            a_max=partial_grid.num_voxels - 1
        ).astype(np.int8)
        step = np.sign(direction).astype(np.int8)
        # t_delta is the distance along t to be traveled to cover one voxel along x
        t_delta = np.abs(partial_grid.voxel_size / direction)  # divison by 0 = inf: good
        # t_max_axis is the distance to be traveled to reach the next axis boundary
        t_max_axis = t_min + (
                min_bound + (current_index + (direction > 0)) *  partial_grid.voxel_size - start
            ) / direction
        return current_index, end_index, step, t_max_axis, t_delta

    # initialize the parameters
    # current_?_index = current voxel index in the ? axis
    # end_?_index = voxel index in the ? axis where the rays exits the grid
    # step_? = {-1, 0, 1} kind of direction of movement in the ? axis
    # t_max_? = distance along t to be traveled to reach the next border parallel to ?
    # t_delta_? = distance along t to be traveled to cross one voxel in the ? axis
    current_x_index, end_x_index, step_x, t_max_x, t_delta_x = init_params(
        ray_start.x, ray_end.x, rays.direction.x, partial_grid.min_bound.x
    )
    current_y_index, end_y_index, step_y, t_max_y, t_delta_y = init_params(
        ray_start.y, ray_end.y, rays.direction.y, partial_grid.min_bound.y
    )
    current_z_index, end_z_index, step_z, t_max_z, t_delta_z = init_params(
        ray_start.z, ray_end.z, rays.direction.z, partial_grid.min_bound.z
    )
    
    # init the mask of rays that are terminated, 
    # i.e. has reached an occupied voxel or end index
    terminated_mask = np.zeros(current_x_index.shape, dtype=bool)

    while True:
        # get the list of rays that have reached an occupied voxel
        occupied_mask = gt_grid.data[
                current_x_index, current_y_index, current_z_index
            ] == VoxelType.occupied
        empty_mask = np.logical_not(occupied_mask)

        # update the grid by setting labels
        partial_grid.data[
            current_x_index[occupied_mask], 
            current_y_index[occupied_mask], 
            current_z_index[occupied_mask]
        ] = VoxelType.occupied

        partial_grid.data[
            current_x_index[empty_mask], 
            current_y_index[empty_mask], 
            current_z_index[empty_mask]
        ] = VoxelType.empty

        # get the list of rays that have reached the end
        end_mask = np.logical_or(
            np.logical_or(
                current_x_index == end_x_index,
                current_y_index == end_y_index
            ),
            current_z_index == end_z_index
        )

        # update the list of terminated
        terminated_mask = np.logical_or(
            terminated_mask, 
            np.logical_or(occupied_mask, end_mask)
        )

        # get the list of remaining rays and check if all have terminated
        remaining_mask = np.logical_not(terminated_mask)
        if remaining_mask.sum() == 0:
            break

        # ?_mask = list of rays that move in ? direction
        x_mask = np.logical_and(
            np.logical_and(t_max_x <= t_max_y, t_max_x <= t_max_z),
            remaining_mask
        )
        y_mask = np.logical_and(
            np.logical_and(t_max_y <= t_max_x, t_max_y <= t_max_z),
            remaining_mask
        )
        z_mask = np.logical_and(
            np.logical_and(t_max_z <= t_max_y, t_max_z <= t_max_y),
            remaining_mask
        )

        # advance the rays
        current_x_index[x_mask] += step_x[x_mask]
        t_max_x[x_mask] += t_delta_x[x_mask]

        current_y_index[y_mask] += step_y[y_mask]
        t_max_y[y_mask] += t_delta_y[y_mask]

        current_z_index[z_mask] += step_z[z_mask]
        t_max_z[z_mask] += t_delta_z[z_mask]