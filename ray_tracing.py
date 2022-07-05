import numpy as np


class VoxelType:
    unseen   = -1
    empty    =  0
    occupied =  1


class Vec3D:

    def __init__(self, np_vector: np.ndarray):
        self.data = np_vector
        self.single = len(self.data.shape) == 1

    @property
    def x(self):
        return self.data[:, 0] if not self.single else self.data[0]

    @property
    def y(self):
        return self.data[:, 1] if not self.single else self.data[1]

    @property
    def z(self):
        return self.data[:, 2] if not self.single else self.data[2]


class Ray:

    def __init__(self, origin: Vec3D, direction: Vec3D):
        """Initializes a ray with a origin point and direction vector.

        Parameters
        ----------
        origin: Vec3D
            (x, y, z) coordinates of the origin point.
        direction: Vec3D
            (x, y, z) coordinates of the direction vector.
        """
        self.origin = origin
        # normalize
        self.direction = Vec3D(
            direction.data / np.linalg.norm(direction.data, axis=-1, keepdims=True)
        )

    
    def __len__(self):
        return self.origin.data.shape[0]


class Grid:

    def __init__(self, voxel_size: float, num_voxels: int, min_bound: Vec3D, max_bound: Vec3D):
        self.voxel_size = voxel_size
        self.num_voxels = num_voxels
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.data = np.ones([num_voxels, num_voxels, num_voxels]) * VoxelType.unseen

    def __idx__(self, idx):
        # TO DO: decide on the specifics of idx
        return self.data[idx]

    def set_label(self, idx, label: VoxelType):
        # TO DO: vectorize
        self.data[idx] = label


def intersection_points(ray, grid):
    """Returns the minimum distance along the ray to enter and exit the 
    grid, t_min and t_max respectively. If there is no intersection,
    returns t_min and t_max are set to inf.
    """
    t_min = np.ones([len(ray),], dtype=np.float32) * -np.inf
    t_max = np.ones([len(ray),], dtype=np.float32) * np.inf

    t_dir_min = np.empty([len(ray),], dtype=np.float32)
    t_dir_max = np.empty([len(ray),], dtype=np.float32)

    # TO DO: what happens if origin is inside the bounds?
    for origin, dir, min_bound, max_bound in zip(
        ray.origin.data.T, ray.direction.data.T,
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


def shoot_rays(ray, grid, max_t_threh, min_t_thresh=0):
    t_min, t_max = intersection_points(ray, grid)
    hits = np.logical_and(t_min != -np.inf, t_max != np.inf)
    t_min = np.clip(t_min, a_min=min_t_thresh, a_max=max_t_threh)
    t_max = np.clip(t_max, a_min=min_t_thresh, a_max=max_t_threh)
    ray_start = Vec3D((ray.origin.data + t_min[:, None] * ray.direction.data)[hits])
    ray_end = Vec3D((ray.origin.data + t_max[:, None] * ray.direction.data)[hits])

    def init_params(start, end, direction, min_bound):
        current_index = np.clip(np.ceil((start - min_bound) / grid.voxel_size), a_min=0, a_max=None)
        end_index = np.clip(np.ceil((end - min_bound) / grid.voxel_size), a_min=0, a_max=None)
        step = np.sign(direction)
        # t_delta is the distance along t to be traveled to cover one voxel in the x direction
        t_delta = np.abs(grid.voxel_size / direction)  # divison by 0 results in inf, which is good
        # t_max_axis is the distance to be traveled to reach the next axis boundary
        t_max_axis = t_min + (
                min_bound + (current_index + (direction > 0)) *  grid.voxel_size - start
            ) / direction

        return current_index, end_index, step, t_max_axis, t_delta

    # initialize the parameters
    # current_?_index = current voxel index in the ? axis
    # end_?_index = voxel index in the ? axis where the ray exits the grid
    # step_? = {-1, 0, 1} kind of direction of movement in the ? axis
    # t_max_? = distance along t to be traveled to reach the next border parallel to ?
    # t_delta_? = distance along t to be traveled to cross one voxel in the ? axis
    current_x_index, end_x_index, step_x, t_max_x, t_delta_x = init_params(
        ray_start.x, ray_end.x, ray.direction.x, grid.min_bound.x
    )
    current_y_index, end_y_index, step_y, t_max_y, t_delta_y = init_params(
        ray_start.y, ray_end.y, ray.direction.y, grid.min_bound.y
    )
    current_z_index, end_z_index, step_z, t_max_z, t_delta_z = init_params(
        ray_start.z, ray_end.z, ray.direction.z, grid.min_bound.z
    )

    while True:
        remaining_mask = np.logical_or(
            np.logical_or(
                current_x_index != end_x_index,
                current_y_index != end_y_index
            ),
            current_z_index != end_z_index
        )
        if remaining_mask.sum() == 0:
            break

        x_mask = np.logical_and(
            np.logical_and(t_max_x < t_max_y, t_max_x < t_max_z),
            remaining_mask
        )
        y_mask = np.logical_and(
            np.logical_and(t_max_y < t_max_x, t_max_y < t_max_z),
            remaining_mask
        )
        z_mask = np.logical_and(
            np.logical_and(t_max_z < t_max_y, t_max_z < t_max_y),
            remaining_mask
        )

        current_x_index[x_mask] += step_x[x_mask]
        t_max_x[x_mask] += t_delta_x[x_mask]

        current_y_index[y_mask] += step_y[y_mask]
        t_max_y[y_mask] += t_delta_y[y_mask]

        current_z_index[z_mask] += step_z[z_mask]
        t_max_z[z_mask] += t_delta_z[z_mask]

        yield np.vstack([current_x_index, current_y_index, current_z_index]).T
        




    