import numpy as np


class VoxelType:
    unseen   : int = -1
    empty    : int =  0
    occupied : int =  1

    @staticmethod
    def is_valid(voxel_type: int):
        return voxel_type in [VoxelType.empty, VoxelType.unseen, VoxelType.occupied]

class Vec3D:
    """A wrapper class for numpy arrays. Holds convenience functions for accessing
    the axes.
    """
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
        """Initializes a ray with an origin point and normalized direction vector.
        """
        self.origin = origin
        # normalize
        self.direction = Vec3D(
            direction.data / np.linalg.norm(direction.data, axis=-1, keepdims=True)
        )

    
    def __len__(self):
        return self.origin.data.shape[0]


class Grid:
    def __init__(
        self, 
        voxel_size: float, 
        num_voxels: int, 
        min_bound: Vec3D, 
        max_bound: Vec3D,
        default_voxel_type: int = VoxelType.unseen
    ):
        assert VoxelType.is_valid(default_voxel_type)
        self.voxel_size = voxel_size
        self.num_voxels = num_voxels
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.data = np.ones([num_voxels, num_voxels, num_voxels]) * default_voxel_type

    
    def set_labels(self, idx: int, label: int):
        assert VoxelType.is_valid(label)
        x, y, z = idx.T
        self.data[x, y, z] = label


    def copy(self):
        copy_grid = Grid(self.voxel_size, self.num_voxels, self.min_bound, self.max_bound)
        copy_grid.data = np.copy(self.data)
        return copy_grid