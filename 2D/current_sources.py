from typing import Any
import numpy as np


def sinusoïdal_point_source(
    q: int,
    M: int,
    pos_x: int,
    pos_y: int,
    total_current: float,
    frequency: float,
    dt: float,
    dx: float,
) -> np.ndarray:
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
        (np.ndarray): 2D array of the current density in [A/m^2]
    """

    J = np.zeros((M, M), dtype=np.float32)

    J[pos_y, pos_x] = total_current / (dx * dx) * np.sin(2 * np.pi * frequency * q * dt, dtype=np.float32)

    return J
