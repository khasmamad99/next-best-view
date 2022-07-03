import numpy as np


"""
Initialization:
Given a ray and a grid, check if the ray intersects with the grid. If it 
does not terminate and return false. If it does, set the ray starting
point to the intersection point.
"""

class VoxelType:
    unseen   = -1
    empty    =  0
    occupied =  1


class Vec3D:

    def __init__(self, np_vector: np.ndarray):
        self.data = np_vector

    @staticmethod
    def normalize(vector):
        data = vector.data / np.linalg.norm(vector.data, axis=1, keepdims=True)
        return Vec3D(data)

    @property
    def x(self):
        return self.data[:, 0]

    @property
    def y(self):
        return self.data[:, 1]

    @property
    def z(self):
        return self.data[:, 2]


class Ray:

    def __init__(self, origin: Vec3D, direction: Vec3D):
        """Initializes a ray with a origining point and direction vector.

        Parameters
        ----------
        origin: Vec3D
            (x, y, z) coordinates of the origining point.
        direction: Vec3D
            (x, y, z) coordinates of the direction vector.
        """
        self.origin = origin
        self.direction = Vec3D.normalize(direction)

    
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
        # To DO: vectorize
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