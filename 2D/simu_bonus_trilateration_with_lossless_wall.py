import numpy as np


from yee_FDTD_2D import (
    compute_electric_field_amplitude,
    compute_electric_field_amplitude_and_plot,
    get_exponential_decay_alpha,
    plot_field,
    using_cupy,
    TYPE_CHECKING,
    xp,
    C_VIDE,
    simulate_and_animate,
    simulate_and_plot,
    u0,
    e0,
)

from simu_elements import (
    sinusoïdal_point_source,
    brick_wall_rel_permittivity,
    create_square,
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from scipy.optimize import minimize


# parameters
Q = 1600  # number of time samples
WIDTH = 13  # [m]
FREQ_REF = 1e9  # [Hz] reference frequency
TOTAL_CURRENT = 0.01  # [A] total current
SOURCE_POS = (WIDTH / 2, WIDTH / 2 - 2)  # [m] position of the source
DETECTOR_1_POS = (WIDTH / 2 - 5, WIDTH / 2)  # [m] position of the detector 1
# DETECTOR_2_POS = (WIDTH / 2, WIDTH / 2 - 5) # [m] position of the detector 2
# DETECTOR_3_POS = (WIDTH/2, WIDTH/2 + 5) # [m] position of the detector 3
WALL_REL_PERMITTIVITY = brick_wall_rel_permittivity  # relative permittivity of the wall
WALL_X_END = WIDTH / 2 - 3  # position of the wall
WALL_THICKNESS = 0.1  # thickness of the wall in meters

# derived parameters
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 40  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
M = round(WIDTH / DELTA_X)  # number of space samples in the y direction
print(f"M : {M}, total number of points : {M * M}")
print(f"wave length : {WAVE_LENGTH}, delta x : {DELTA_X}, delta t : {DELTA_T}")
print(
    f"actal source position : {round(SOURCE_POS[0] / DELTA_X) * DELTA_X}, {round(SOURCE_POS[1] / DELTA_X) * DELTA_X}"
)

speed_of_light_in_wall = np.sqrt(1 / (u0 * e0 * brick_wall_rel_permittivity))
print(
    f"number of time steps per wavelength in wall : {speed_of_light_in_wall / (DELTA_X * FREQ_REF):.2f}"
)


# constants
Z_0 = np.sqrt(u0 / e0)
beta = 2 * np.pi * FREQ_REF / C_VIDE
Z_2 = np.sqrt(u0 / (e0 * WALL_REL_PERMITTIVITY))  # characteristic impedance in the wall


def get_reflexion_coeff(Z1: float, Z2: float, theta_i, theta_t) -> float:
    return (Z2 * np.cos(theta_i) - Z1 * np.cos(theta_t)) / (
        Z2 * np.cos(theta_i) + Z1 * np.cos(theta_t)
    )

def get_transmission_coeff(Z1: float, Z2: float, theta_i, theta_t) -> float:
    return (2 * Z2 * np.cos(theta_i)) / (Z2 * np.cos(theta_i) + Z1 * np.cos(theta_t))

def source_func(q: int, current_J) -> None:
    sinusoïdal_point_source(
        current_J,
        q,
        M,
        round(SOURCE_POS[0] / DELTA_X),
        round(SOURCE_POS[1] / DELTA_X),
        TOTAL_CURRENT,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
    )


wall_rel_perm = create_square(
    (WALL_X_END, 0),
    WALL_THICKNESS,
    WIDTH,
    (M, M),
    DELTA_X,
    value=WALL_REL_PERMITTIVITY,
    default_value=1.0,
)

E_amp = compute_electric_field_amplitude(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1 / FREQ_REF,
    source_func,
    local_rel_permittivity=wall_rel_perm,
)

figure, ax = plt.subplots(1, 1, figsize=(6, 6))

plot_field(
    ax,
    DELTA_X,
    E_amp,
)


def distance_from_source(E_amplitude: float) -> float:
    return Z_0**2 * (beta / (8 * np.pi)) * TOTAL_CURRENT**2 / (E_amplitude**2)


def distance_from_source_through_wall(E_amplitude: float) -> float:
    """Calculate the distance from the source to the detector through the wall."""

    pass


def add_detector(
    ax: Axes,
    pos: tuple[float, float],
    color: str,
    E_amp,
    distance_func=distance_from_source,
) -> None:
    """Add a detector to the plot."""
    # Show the position of the detectors
    detector_pos = Circle(
        (round(pos[0] / DELTA_X), round(pos[1] / DELTA_X)),
        M / 100,
        color=color,
        fill=True,
    )
    ax.add_patch(detector_pos)

    # Show the distance from the source to the detector
    distance = distance_func(E_amp[round(pos[1] / DELTA_X), round(pos[0] / DELTA_X)])
    print(f"distance from source to detector : {distance}")
    circle = Circle(
        (round(pos[0] / DELTA_X), round(pos[1] / DELTA_X)),
        distance / DELTA_X,
        color=color,
        fill=False,
        linestyle="--",
        linewidth=2.5,
    )
    ax.add_patch(circle)


add_detector(ax, DETECTOR_1_POS, "red", E_amp)
# add_detector(ax, DETECTOR_2_POS, "blue", E_amp)
# add_detector(ax, DETECTOR_3_POS, "green", E_amp)


# detects_list = [
#     DETECTOR_1_POS,
#     DETECTOR_2_POS,
#     DETECTOR_3_POS,
# ]
# detector_estimated_distance = [
#     distance_from_source(E_amp[round(DETECTOR_1_POS[1]/DELTA_X), round(DETECTOR_1_POS[0]/DELTA_X)]),
#     distance_from_source(E_amp[round(DETECTOR_2_POS[1]/DELTA_X), round(DETECTOR_2_POS[0]/DELTA_X)]),
#     distance_from_source(E_amp[round(DETECTOR_3_POS[1]/DELTA_X), round(DETECTOR_3_POS[0]/DELTA_X)]),
# ]
# # find estimed position of the source
# def cost_fucntion(X:np.ndarray) -> float:
#     """Cost function to minimize."""
#     # X[0] = x
#     # X[1] = y
#     total_cost = 0
#     for detector_pos, distance in zip(detects_list, detector_estimated_distance):
#         # distance from the source to the detector
#         distance_from_source = np.sqrt((X[0] - detector_pos[0])**2 + (X[1] - detector_pos[1])**2)
#         total_cost += (distance_from_source - distance)**2
#     return total_cost


# x_0, y_0 = WIDTH / 2, WIDTH / 2

# x_source_est, y_source_est = minimize(
#     cost_fucntion,
#     x0=np.array([x_0, y_0]),
#     ).x
# print(f"estimated source position : {x_source_est}, {y_source_est}")
# print(f"actual source position : {SOURCE_POS[0]}, {SOURCE_POS[1]}")
# print(f"estimation error : {np.sqrt((x_source_est - SOURCE_POS[0])**2 + (y_source_est - SOURCE_POS[1])**2)}")

plt.show()
