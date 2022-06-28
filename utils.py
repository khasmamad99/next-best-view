import numpy as np
import open3d as o3d
import k3d


def visualize_occupancy(occupancy_grid, do_flip_axes=False):
    if not isinstance(occupancy_grid, (list, tuple)):
        occupancy_grid =  [occupancy_grid]
    points_list = list()
    for og in occupancy_grid:
        points = np.concatenate([c[:, np.newaxis] for c in np.where(og)], axis=1)
        points_list.append(points)
    visualize_pointcloud(points_list, 1, do_flip_axes, name='occupancy_grid')


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


def spherical2cartesian(rtp):
    r, t, p = rtp
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return np.array([x, y, z])


def cartesian2spherical(xyz):
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x) % (2 * np.pi) # change output range to 0 - 2pi
    return np.array([r, theta, phi])


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([
        [1, 0, 0], 
        [0, np.cos(r_x), -np.sin(r_x)],
        [0, np.sin(r_x), np.cos(r_x)]
    ])
    rot_y = np.asarray([
        [np.cos(r_y), 0, np.sin(r_y)], 
        [0, 1, 0],
        [-np.sin(r_y), 0, np.cos(r_y)]
    ])
    return rot_y.dot(rot_x)


def get_extrinsic_cam_mtx(cam_posisition_real_world):
    spherical_coords = cartesian2spherical(cam_posisition_real_world)
    rot_mtx = get_rotation_matrix(spherical_coords[1], spherical_coords[2])  # rotation matrix (world to camera)
    trans_mtx = np.asarray([0, 0, spherical_coords[0]]).transpose() # world origin in camera coords
    extrinsic_cam_mtx = np.eye(4)
    extrinsic_cam_mtx[:3, :3] = rot_mtx
    extrinsic_cam_mtx[:3, 3] = trans_mtx
    return extrinsic_cam_mtx


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
    
    Apparently, open3d orders axes in a different way. This function
    reorders the axes in the canonical way.
    """
    points[:, 2] = points[:, 2] * -1
    points[:, [0, 1, 2]] = points[:, [0, 2, 1]]
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
    return occupancy_grid


def load_candidate_views(unit_sphere_radius, path='candidate_views.txt'):
    """Loads candidate views from the given path and scales them
    so that they lie on a sphere with a radius of unit_sphere_radius.
    """
    nbv_positions = np.genfromtxt(path)
    nbv_positions = unit_sphere_radius * nbv_positions / np.linalg.norm(nbv_positions, axis=1, keepdims=True) 
    return nbv_positions


def setup_virtual_camera(width, height):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    return vis, ctr, param


def capture_depth_image(view, vis, ctr, param):
    """View is real world cartesian coordinates of the camera center.
    """
    param.extrinsic = get_extrinsic_cam_mtx(view)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer(False)
    depth = np.asarray(depth)
    return depth


def depthimage2pointcloud(depth, view, param):
    param.extrinsic = get_extrinsic_cam_mtx(view)
    depth_obj = o3d.geometry.Image(depth)
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_obj, param.intrinsic, param.extrinsic, depth_scale=1
    )
    return point_cloud



   
