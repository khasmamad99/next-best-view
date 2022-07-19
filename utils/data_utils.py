import numpy as np
import open3d as o3d


def init_camera_intrinsic(camera: str = "kinect"):
    camera = camera.lower()
    assert camera in ["kinect", "primesense"]
    if camera == "kinect":
        return o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
    elif camera == "primsense":
        return o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)


def random_sphere_point(npoints: int=1, ndim: int=3, radius: float=1.):
    points = np.random.randn(npoints, ndim)
    points /= np.linalg.norm(points, axis=-1)
    points *= radius
    return points


def spherical2cartesian(rtp):
    if len(rtp.shape) == 1:
        r, t, p = rtp
    else:
        r = rtp[:, 0]
        t = rtp[:, 1]
        p = rtp[:, 2]

    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return np.hstack([x, y, z])


def cartesian2spherical(xyz):
    if len(xyz.shape) == 1:
        x, y, z = xyz
    else:
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

    r = np.linalg.norm(xyz, axis=-1)
    theta = np.arccos(z / r)
    # calculate phi and change output range to 0 - 2pi
    phi = np.arctan2(y, x) % (2 * np.pi)
    return np.hstack([r, theta, phi])


def setup_virtual_camera(width, height):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    return vis, ctr, param


def normalize(points):
    """Given a list of points in 3D, centers them at (0, 0, 0)
    and rescales them so that they lie within the unit circle.
    """
    max_bound = np.max(np.abs(points), axis=0)
    center = points.mean(axis=0)
    scale = np.linalg.norm(max_bound - center)
    points -= center
    points /= scale
    return points


def flip_axes(points):
    """Inverts the z axis and then exchanges y and z axes.
    
    There is a mismatch between how open3d vs k3d orders the axes. This is
    a remedy for that
    """
    points = np.copy(points)
    single_point = False
    if len(points.shape) == 1:
        points = points[None, :]
        single_point = True
    points[:, 2] = points[:, 2] * -1
    points[:, [0, 1, 2]] = points[:, [0, 2, 1]]
    if single_point:
        points = points[0]
    return points


def to_occupancy_grid(
    model, 
    input_type="mesh", # or "point_cloud"
    do_normalization=False,   
    num_voxels=32, 
):
    """Creates a num_voxels x num_voxels x num_voxels binary occupancy grid from a given
    model (mesh or points cloud). Assumes that the given model is within the unit sphere.
    """
    assert input_type in ["mesh", "point_cloud"]
    np_points = np.asarray(model.vertices) if input_type == "mesh" \
                                           else np.asarray(model.points)
    if do_normalization:
        np_points = normalize(np_points)
        if input_type == "mesh":
            model.vertices = o3d.utility.Vector3dVector(np_points)
        elif input_type == "point_cloud":
            model.points = o3d.utility.Vector3dVector(np_points)
    
    # assumes that the model is within the unit sphere
    voxel_size = 2. / num_voxels
    max_bound = np.array([1., 1., 1.])
    min_bound = -max_bound
    if input_type == "mesh":
        voxel_grid_obj = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            model, voxel_size, min_bound, max_bound
        )
    elif input_type == "point_cloud":
        voxel_grid_obj = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            model, voxel_size, min_bound, max_bound
        )
    occupied_voxel_indices = np.asarray(list(
        map(lambda x: x.grid_index, voxel_grid_obj.get_voxels()))
    )
    occupied_voxel_indices = tuple(occupied_voxel_indices.T.tolist())
    occupancy_grid = np.zeros([num_voxels, num_voxels, num_voxels])
    occupancy_grid[occupied_voxel_indices] = 1
    return occupancy_grid, voxel_size, min_bound, max_bound



