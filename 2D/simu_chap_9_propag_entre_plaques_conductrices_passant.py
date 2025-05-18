import numpy as np
import cupy as cp

from yee_FDTD_2D import (
    compute_electric_field_amplitude_and_plot,
    get_exponential_decay_alpha,
    plot_field,
    using_cupy,
    TYPE_CHECKING,
    xp,
    C_VIDE,
    simulate_and_animate,
    simulate_and_plot,
    simulate_up_to,
)

from simu_elements import (
    sinusoïdal_point_source,
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

Q = 5000  # number of time samples
WIDTH = 0.1  # [m] distance between the plates
LENGTH = 7  # [m] length of the plates
FREQ_REF = 2.5e9  # [Hz] reference frequency
TOTAL_CURRENT = 0.0005  # [A] total current
SOURCE_X_POS = 0.2  # [m] position of the source

WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 100  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
N = round(LENGTH / DELTA_X)  # number of space samples in the x direction
M = round(WIDTH / DELTA_X)  # number of space samples in the y direction
print(f"N : {N}, M : {M}, total number of points : {N * M}")
print(f"wave length : {WAVE_LENGTH}, delta x : {DELTA_X}, delta t : {DELTA_T}")

# Phase array parameters
# simple source : https://www.youtube.com/watch?v=jSDLfcNhThw
NUMBER_OF_ELEMENTS = int(M)  # number of elements in the phased array
ELEMENT_SPACING = DELTA_X  # [m]
ELEMENT_SPACING_INDEX = int(
    ELEMENT_SPACING / DELTA_X
)  # number of grid points between elements
TARGET_ANGLE = 53.1  # [degrees]


# def phase_distribution(index: int) -> float:
#     return (
#         np.pi
#         * index
#         * ELEMENT_SPACING
#         * np.sin(TARGET_ANGLE * np.pi / 180)
#         / WAVE_LENGTH
#     )
start_index = 0


def current_func(q: int, previous_J):
    previous_J[
        start_index + 0 : NUMBER_OF_ELEMENTS : ELEMENT_SPACING_INDEX,
        1,
    ] = (
        TOTAL_CURRENT
        / (DELTA_X * DELTA_X)
        * xp.sin(
            2 * xp.pi * FREQ_REF * q * DELTA_T
            + (
                xp.pi
                * xp.arange(0, NUMBER_OF_ELEMENTS, ELEMENT_SPACING_INDEX)
                * ELEMENT_SPACING
                * xp.sin(TARGET_ANGLE * xp.pi / 180)
                / WAVE_LENGTH
            ),
            dtype=xp.float32,
        )
        * (1 - xp.exp(-q * DELTA_T / (3 / FREQ_REF), dtype=xp.float32))
    )


print(
    f"final exponential smoothing factor : {1 - xp.exp(-Q * DELTA_T / (3 / FREQ_REF), dtype=xp.float32)}"
)

# E0 = np.ones((M, N), dtype=np.float32) * 1e-30
# B_tilde_0 = np.ones((M, N), dtype=np.float32) * 1e-30

# simulate_and_animate(
#     E0,
#     B_tilde_0,
#     DELTA_T,
#     DELTA_X,
#     min_color_value=1,
#     max_color_value=40,
#     q_max=Q,
#     m_max=M,
#     n_max=N,
#     current_func=current_func,
#     show_material=False,
#     norm_type="lin",
# )


fig, ax1 = plt.subplots(1, 1, figsize=(10, 3))
E, _, _ = simulate_up_to(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    current_func,
    n_max=N,
    # min_color_value=2e-2,
    return_numpy=True,
)
im = plot_field(
    ax1,
    DELTA_X,
    E[:, 0 : round(0.5  / DELTA_X)],
    norm_type="lin",
)


plt.title("Valeur instantanée du champ électrique")

plt.savefig("images/propagation_entre_plaques_conductrices_passant.png", dpi=300)
fig, ax2 = plt.subplots(1, 1, figsize=(6,6))
ax2.plot(np.linspace(0, 0.5, E[M//2, 0 : round(0.5 / DELTA_X)].shape[0]), E[M//2, 0 : round(0.5 / DELTA_X)], label="Champ électrique")
plt.grid()
plt.show()
