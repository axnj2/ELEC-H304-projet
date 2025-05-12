import numpy as np

from yee_FDTD_2D import simulate_and_animate, e0, C_VIDE
from current_sources import sinusoïdal_point_source

# parameters
# settings parameters
microwave_side_length = 0.357  # in meters
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
