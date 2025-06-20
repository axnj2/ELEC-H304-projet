import numpy as np

from yee_FDTD_2D import simulate_and_animate, e0, C_VIDE
from simu_elements import sinusoïdal_point_source


# parameters
# settings parameters
M = 401  # number of space samples per dimension
FREQ_REF = 1e8  # Hz
Q = 500  # number of time samples
TOTAL_CURRENT = 0.01  # A
INITIAL_ZERO = 0  # initial value for E and B_tilde
MIN_COLOR = 1e-4  # minimum color value for the image


# derived parameters
DELTA_X = C_VIDE / (FREQ_REF * 20)  # in meters
DELTA_T = 1 / (2 * FREQ_REF * 20)  # in seconds
all_time_max = TOTAL_CURRENT / (DELTA_X * DELTA_X) * DELTA_T / e0


def current_func(q: int, current_J) -> None:
    sinusoïdal_point_source(
        current_J,
        q,
        M,
        M // 2,
        M // 2,
        TOTAL_CURRENT,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
    )


# initialise the starting values
E0 = np.ones((M, M)) * INITIAL_ZERO
B_tilde_0 = np.ones((M, M)) * INITIAL_ZERO
perfect_conductor = np.zeros((M, M), dtype=np.bool)
perfect_conductor[M // 4, :] = True

simulate_and_animate(
    E0,
    B_tilde_0,
    DELTA_T,
    DELTA_X,
    MIN_COLOR,
    all_time_max,
    Q,
    M,
    current_func,
    norm_type="log",
    perfect_conductor_mask=perfect_conductor,
    use_progress_bar=True,
)
