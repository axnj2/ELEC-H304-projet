import numpy as np

from typing import TYPE_CHECKING
from yee_FDTD_2D import simulate_and_plot, e0, u0, C_VIDE, xp, using_cupy
from simu_elements import sinusoïdal_point_source

import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.axes import Axes


# parameters
# settings parameters
M = 300  # number of space samples per dimension
FREQ_REF = 1e8  # Hz
Q = int(M / 1.2)  # number of time steps
TOTAL_CURRENT = 0.01  # A


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


TOTAL_X = (M - 1) * DELTA_X  # in meters
TOTAL_T = (Q - 1) * DELTA_T  # in seconds
print("TOTAL_X : ", TOTAL_X, "TOTAL_T : ", TOTAL_T)
print("DELTA_X : ", DELTA_X, "DELTA_T : ", DELTA_T)
print("C_VIDE : ", C_VIDE)


# plot E as an image
return_value = plt.subplots(1, 2, figsize=(14, 6))
fig = return_value[0]
ax1: Axes = return_value[1][0]
ax2: Axes = return_value[1][1]

im, E = simulate_and_plot(ax1, DELTA_T, DELTA_X, Q, M, current_func, norm_type="lin")

E_max = np.max(np.abs(E))

ax2.plot(
    np.abs(E[M // 2, :]),
    np.linspace(0, TOTAL_X, M) - TOTAL_X / 2,
    label=f"simulated electric field",
)
ax2.set_ylabel("r (m)")
ax2.set_xlabel("|Ez| (V/m)")
ax2.set_title(f"Electric field E at t = {DELTA_T * Q:.2e} s")

# add the theoritical value
Z_0 = np.sqrt(u0 / e0)
beta = 2 * np.pi * FREQ_REF / C_VIDE
x_axis = np.linspace(0, TOTAL_X / 2, M // 2)
# make it symmetric around the center
x_axis = np.concatenate((x_axis[::-1], x_axis))

E_theoritical = TOTAL_CURRENT * Z_0 * np.sqrt(beta / (8 * np.pi)) / (np.sqrt(x_axis))

ax2.plot(
    E_theoritical,
    np.linspace(0, TOTAL_X, (M // 2) * 2) - TOTAL_X / 2,
    label="Theoretical value",
    linestyle="--",
)
ax2.legend()

# find the position of the non-zero values

non_zero = np.abs(E[M // 2, :]) > 0.01 * E_max
index = np.where(non_zero)[0]
distance = (M // 2 - index[0]) * DELTA_X
print("distance travalled: ", distance, "m")
print("time travalled: ", DELTA_T * (Q), "s")
print("speed: ", distance / (DELTA_T * (Q)), "m/s")
print("expected speed: ", C_VIDE, "m/s")
print("relative error: ", (distance / (DELTA_T * (Q)) - C_VIDE) / C_VIDE)


plt.savefig("images/current_line_validation.png", bbox_inches="tight")
plt.show()
