from typing import TYPE_CHECKING

import numpy as xp
import numpy as np

import math

if TYPE_CHECKING:
    pass
else:
    try:
        import cupy

        if cupy.cuda.is_available():
            xp = cupy

    except ImportError:
        pass


# Constants
brick_wall_rel_permittivity = 4.0  # TODO find a realistic value
brick_wall_conductivity = 0.01  # S/m TODO find a realistic value
wooden_door_rel_permittivity = 2.0  # TODO find a realistic value
wooden_door_conductivity = 0.001  # S/m TODO find a realistic value

# from https://doi.org/10.1016/j.jfoodeng.2011.05.031
# at 20°C and 1800MHz
meat_relative_real_permittivity = 40.1
meat_relative_complex_permittivity = 14.3
cheese_relative_real_permittivity = 25.6
cheese_relative_complex_permittivity = 11.9
pasta_relative_real_permittivity = 50.9
pasta_relative_complex_permittivity = 19.6
sauce_relative_real_permittivity = 76.4
sauce_relative_complex_permittivity = 27.6

# thermal_caracteristics


def sinusoïdal_point_source(
    previousJ: xp.ndarray,
    q: int,
    M: int,
    pos_x: int,
    pos_y: int,
    total_current: float,
    frequency: float,
    dt: float,
    dx: float,
    phase: float = 0.0,
) -> None:
    """Generate a sinusoidal point source with a total given current at [pos_x, pos_y] in the grid.

    Args:
        q (int): time step number
        M (int): grid size
        pos_x (int): x index of the point source
        pos_y (int): y index of the point source
        total_current (float): total current in [A]
        frequency (float): frequency in [Hz]
        dt (float): time step in [s]
        dx (float): space step in [m]

    Returns:
        (xp.ndarray): 2D array of the current density in [A/m^2]
    """
    previousJ[pos_y, pos_x] = (
        total_current
        / (dx * dx)
        * xp.sin(2 * np.pi * frequency * q * dt + phase, dtype=xp.float32)
    )


def create_square_boundery(
    upper_left_corner: np.ndarray | tuple[float, float],
    x_size: float,
    y_size: float,
    grid_size: tuple[int, int],
    delta_x: float,
    value: float | bool,
    default_value: float | bool,
    thickness: float | None = None,
) -> xp.ndarray:
    """Create a square boundery with a given size, position and material properties.
    Only one of the three material properties can be set at a time.

    Args:
        upper_left_corner (np.ndarray):
        x_size (float):
        y_size (float):
        grid_size (tuple[int, int]):
        delta_x (float):
        value (float | bool): material property to set
        default_value (float | bool): value to set the rest of the grid to
            If value is a float, default_value must be a float, and vice versa.
        thickness (float | None, optional): if it is None, will default to 1 space step thick. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    match [value, default_value]:
        case float(), float():
            if value < 0:
                raise ValueError("value must be positive")

            boundery = xp.ones(grid_size, dtype=np.float32) * default_value
        case bool(), bool():
            boundery = xp.ones(grid_size, dtype=np.bool) * default_value
        case _:
            raise TypeError(
                "value must be a float or a bool, and coherent with default_value"
            )

    if thickness is None:
        discrete_thickness: int = 1
    else:
        discrete_thickness: int = math.ceil(thickness / delta_x)
        realised_thickness: float = discrete_thickness * delta_x
        error_on_thickness = abs(realised_thickness - thickness) / thickness
        if error_on_thickness > 0.2:
            Warning(
                f"the grid did not allow to create a boundery of {thickness} m,  a boundery of {realised_thickness} m was created instead, error on thickness: {error_on_thickness:.1%}"
            )

    # support for negative widths and heights
    if x_size < 0:
        x_size = -x_size
        upper_left_corner = np.array(
            (upper_left_corner[0] - x_size, upper_left_corner[1])
        )
    if y_size < 0:
        y_size = -y_size
        upper_left_corner = np.array(
            (upper_left_corner[0], upper_left_corner[1] - y_size)
        )

    # create the boundery
    x_start = math.floor(upper_left_corner[0] / delta_x)
    y_start = math.floor(upper_left_corner[1] / delta_x)
    x_end = math.ceil((upper_left_corner[0] + x_size) / delta_x)
    y_end = math.ceil((upper_left_corner[1] + y_size) / delta_x)
    boundery[y_start:y_end, x_start:x_end] = value

    assert x_end - x_start - 2 * discrete_thickness > 0
    assert y_end - y_start - 2 * discrete_thickness > 0

    # add the thickness
    boundery[
        y_start + discrete_thickness : y_end - discrete_thickness,
        x_start + discrete_thickness : x_end - discrete_thickness,
    ] = default_value

    return boundery


def create_square(
    upper_left_corner: np.ndarray | tuple[float, float],
    x_size: float,
    y_size: float,
    grid_size: tuple[int, int],
    delta_x: float,
    value: float | bool,
    default_value: float | bool,
) -> xp.ndarray:
    """Create a square boundery with a given size, position and material properties.
    Only one of the three material properties can be set at a time.

    Args:
        upper_left_corner (np.ndarray):
        x_size (float):
        y_size (float):
        grid_size (tuple[int, int]):
        delta_x (float):
        value (float | bool): material property to set
        default_value (float | bool): value to set the rest of the grid to
            If value is a float, default_value must be a float, and vice versa.
        thickness (float | None, optional): if it is None, will default to 1 space step thick. Defaults to None.

    Returns:
        xp.ndarray: _description_
    """
    match [value, default_value]:
        case float(), float():
            if value < 0:
                raise ValueError("value must be positive")

            square = xp.ones(grid_size, dtype=np.float32) * default_value
        case bool(), bool():
            square = xp.ones(grid_size, dtype=np.bool) * default_value
        case _:
            raise TypeError(
                "value must be a float or a bool, and coherent with default_value"
            )

    # support for negative widths and heights
    if x_size < 0:
        x_size = -x_size
        upper_left_corner = np.array(
            (upper_left_corner[0] - x_size, upper_left_corner[1])
        )
    if y_size < 0:
        y_size = -y_size
        upper_left_corner = np.array(
            (upper_left_corner[0], upper_left_corner[1] - y_size)
        )

    x_start = math.floor(upper_left_corner[0] / delta_x)
    y_start = math.floor(upper_left_corner[1] / delta_x)
    x_end = math.ceil((upper_left_corner[0] + x_size) / delta_x)
    y_end = math.ceil((upper_left_corner[1] + y_size) / delta_x)
    square[y_start:y_end, x_start:x_end] = value

    return square
