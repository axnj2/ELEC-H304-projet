import numpy as np

from yee_FDTD_2D import simulate_and_animate, e0, C_VIDE
from simu_elements import sinusoïdal_point_source

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
REFINEMENT_FACTOR = 40 # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
all_time_max = TOTAL_CURRENT / (DELTA_X * DELTA_X) * DELTA_T / e0

# Phase array parameters
# simple source : https://www.youtube.com/watch?v=jSDLfcNhThw
NUMBER_OF_ELEMENTS = 90  # number of elements in the phased array
ELEMENT_SPACING = WAVE_LENGTH / 5 # [m]
ELEMENT_SPACING_INDEX = int(ELEMENT_SPACING / DELTA_X)  # number of grid points between elements
TARGET_ANGLE = 180  # [degrees]

def phase_distribution(index: int) -> float:

    return np.pi * index * ELEMENT_SPACING * np.sin(TARGET_ANGLE * np.pi / 180) / WAVE_LENGTH

def current_func(q: int, previous_J):
    start_index = int(M / 2) - ELEMENT_SPACING_INDEX * (NUMBER_OF_ELEMENTS - 1) // 2
    for ii in range(NUMBER_OF_ELEMENTS):
        sinusoïdal_point_source(
            previous_J,
            q,
            M,
            start_index + ii * ELEMENT_SPACING_INDEX,
            int(M / 2),
            TOTAL_CURRENT,
            FREQ_REF,
            DELTA_T,
            DELTA_X,
            phase=phase_distribution(
                ii
            ),
        )

# initialise the starting values
E0 = np.ones((M, M), dtype=np.float32) * INITIAL_ZERO
B_tilde_0 = np.ones((M, M), dtype=np.float32) * INITIAL_ZERO
J0 = np.zeros((M, M), dtype=np.float32)

simulate_and_animate(
    E0,
    B_tilde_0,
    DELTA_T,
    DELTA_X,
    MIN_COLOR,
    all_time_max / 2,
    Q,
    M,
    current_func=current_func,
    norm_type="log",
    use_progress_bar=True,
    precompute=False,
    min_time_per_frame=0,
)
