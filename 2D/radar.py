import numpy as np

from yee_FDTD_2D import simulate_and_animate, e0, C_VIDE
from current_sources import sinusoïdal_point_source

# parameters
# settings parameters
M = 1500  # number of space samples per dimension
FREQ_REF = 1e8  # Hz
Q = 1000  # number of time samples
TOTAL_CURRENT = 0.01  # A
INITIAL_ZERO = 0  # initial value for E and B_tilde
MIN_COLOR = 1e-1  # minimum color value for the image


# derived parameters
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 40  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
all_time_max = TOTAL_CURRENT / (DELTA_X * DELTA_X) * DELTA_T / e0


# create a conductive target
def create_square_target(
    upper_left_corner: tuple[int, int],
    x_size: int,
    y_size: int,
    grid_size: tuple[int, int],
) ->  np.ndarray:
    """
    Create a perfectly conductive square target.


    Args:
        upper_left_corner (tuple[int, int]):
        x_size (int): 
        y_size (int): 

    Returns:
        np.ndarray: 
    """
    