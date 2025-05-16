from matplotlib import pyplot as plt
import numpy as np

from yee_FDTD_2D import simulate_and_plot, e0, C_VIDE
from simu_elements import sinusoïdal_point_source


# parameters
# settings parameters
M = 1001  # number of space samples per dimension
FREQ_REF = 1e8  # Hz
Q = 1000  # number of time samples
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
local_conductivity = np.zeros((M, M), dtype=np.float32)
local_conductivity[0 : M // 4, :] = 0.003

fig, ax = plt.subplots()

im = simulate_and_plot(
    ax,
    DELTA_T,
    DELTA_X,
    Q,
    M,
    current_func,
    min_color_value=MIN_COLOR,
    norm_type="log",
    local_conductivity=local_conductivity,
)




plt.show()