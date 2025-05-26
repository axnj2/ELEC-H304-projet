# FDTD - Finite-Difference Time-Domain
# 2D simulation using Yee's algorithm
# to simulate electromagnetic waves.

import numpy as np

from yee_FDTD_2D import simulate_and_animate, e0, C_VIDE, simulate_and_plot
from simu_elements import gaussian_point_source

from matplotlib import pyplot as plt

# parameters
# settings parameters
M = 2001  # number of space samples per dimension
GAUSSIAN_SIGMA = 1e-9
alpha = 1 / (2 * GAUSSIAN_SIGMA**2)
FREQ_REF = np.sqrt(1 * alpha) * 2 * np.pi  # [Hz] reference frequency
print(f"FREQ_REF : {FREQ_REF:.2e} Hz")
Q = 20000  # number of time samples
TOTAL_CURRENT = 0.01  # A
INITIAL_ZERO = 0  # initial value for E and B_tilde
MIN_COLOR = 1e-5  # minimum color value for the image


# derived parameters
DELTA_X = C_VIDE / (FREQ_REF * 20)  # in meters
DELTA_T = 1 / (2 * FREQ_REF * 20)  # in seconds
all_time_max = TOTAL_CURRENT / (DELTA_X * DELTA_X) * DELTA_T / e0


def current_func(q: int, current_J) -> None:
    gaussian_point_source(
        current_J,
        q,
        M,
        M // 2,
        M // 2,
        TOTAL_CURRENT,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
        sigma=GAUSSIAN_SIGMA,
    )


TOTAL_X = (M - 1) * DELTA_X  # in meters
TOTAL_T = (Q - 1) * DELTA_T  # in seconds
print("TOTAL_X : ", TOTAL_X, "TOTAL_T : ", TOTAL_T)
print("DELTA_X : ", DELTA_X, "DELTA_T : ", DELTA_T)


# initialise the starting values
E0 = np.ones((M, M), dtype=np.float32) * INITIAL_ZERO
B_tilde_0 = np.ones((M, M), dtype=np.float32) * INITIAL_ZERO


simulate_and_animate(
    E0,
    B_tilde_0,
    DELTA_T,
    DELTA_X,
    MIN_COLOR,
    all_time_max/200,
    Q,
    M,
    current_func,
    norm_type="lin",
    use_progress_bar=True,
    precompute=False,
    min_time_per_frame=10,
)

# all_J = []
# currentJ = np.zeros((M, M), dtype=np.float32)
# for q in range(Q):
#     current_func(q, currentJ)
#     all_J.append(currentJ[M//2, M//2])

# plt.plot(all_J)
# plt.show()