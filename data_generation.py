import numpy as np
from utils import cartesian2spherical, spherical2cartesian


# a full circle is traversed in STEP_FACTOR * 2 steps (see STEP_SIZE)
STEP_FACTOR = 6
PI = np.pi
STEP_SIZE = PI / STEP_FACTOR  # in radians


def move(direction, location):
    """ Calculates the new location after moving from the given location in the given
    direction.

    Movement happens on a sphere. Up, down, left, and right correspond to north,
    south, east, and west. Up/down movements happen along the longitude,
    while left/right movements happen along the latitude. If close to a pole,
    left/right movements are restricted to one-third of the perimeter of the
    current latitude circle to avoid moving full-circle. Up/down movements
    can go across a pole. Since there is no direction at the poles,
    movement does not stop at a pole.

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
            theta -= 0.1 * STEP_SIZE
    elif direction == "down":
        theta += STEP_SIZE
        # check if at the pole, if so move further
        if np.isclose(theta, PI, atol=1e-2):
            theta += 0.1 * STEP_SIZE

    # check if moved across a pole and changed hemisphere
    if theta < 0 or theta > PI:
        theta = np.abs(theta) % PI
        phi = (phi + PI) % (2 * PI)  # shift 180 degrees and change hemisphere
    
    if direction == "left" or direction == "right":
        step_arc_length = STEP_SIZE * r
        latitude_radius = r * np.sin(theta)
        latitude_perimeter = 2 * PI * latitude_radius
        step_size = min(step_arc_length, latitude_perimeter / 3)  # restrict to one-third
        step_size_radians = step_size / latitude_radius
        print(step_size_radians, latitude_perimeter, step_size)
        if direction == "left":
            phi -= step_size_radians
        else:
            phi += step_size_radians
        phi %= (2 * PI) # fix the range back to [0, 2PI]
    
        print(r, theta, phi)

    new_location = spherical2cartesian([r, theta, phi])
    return new_location







    


    

