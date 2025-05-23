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
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle



# parameters
Q = 1500  # number of time samples
WIDTH = 25  # [m] 
FREQ_REF = 1e9  # [Hz] reference frequency
TOTAL_CURRENT = 0.01  # [A] total current
SOURCE_POS = (WIDTH / 2 - 2, WIDTH / 2 - 2) # [m] position of the source
DETECTOR_1_POS = (WIDTH / 2 - 5, WIDTH / 2) # [m] position of the detector 1
DETECTOR_2_POS = (WIDTH / 2, WIDTH / 2 - 5) # [m] position of the detector 2
DETECTOR_3_POS = (WIDTH/2, WIDTH/2 + 5) # [m] position of the detector 3

# derived parameters
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 20  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
M = round(WIDTH / DELTA_X)  # number of space samples in the y direction
print(f"M : {M}, total number of points : {M * M}")
print(f"wave length : {WAVE_LENGTH}, delta x : {DELTA_X}, delta t : {DELTA_T}")
print(f"actal source position : {round(SOURCE_POS[0]/DELTA_X) * DELTA_X}, {round(SOURCE_POS[1]/DELTA_X) * DELTA_X}")

# constants
Z_0 = np.sqrt(u0 / e0)
beta = 2 * np.pi * FREQ_REF / C_VIDE

def source_func(q: int, current_J) -> None:
    sinusoïdal_point_source(
        current_J,
        q,
        M,
        round(SOURCE_POS[0]/DELTA_X),
        round(SOURCE_POS[1]/DELTA_X),
        TOTAL_CURRENT,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
    )

E_amp = compute_electric_field_amplitude(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1/FREQ_REF,
    source_func,
)

figure, ax = plt.subplots(1, 1, figsize=(6, 6))

plot_field(
    ax,
    DELTA_X,
    E_amp,
)

def distance_from_source(E_amplitude: float) -> float:
    return Z_0**2 * (beta/(8*np.pi)) * TOTAL_CURRENT**2 / (E_amplitude**2)

distance_from_detector_1 = distance_from_source(E_amp[round(DETECTOR_1_POS[1]/DELTA_X), round(DETECTOR_1_POS[0]/DELTA_X)])
distance_from_detector_2 = distance_from_source(E_amp[round(DETECTOR_2_POS[1]/DELTA_X), round(DETECTOR_2_POS[0]/DELTA_X)])
print(f"distance from source to detector 1 : {distance_from_detector_1}")
print(f"distance from source to detector 2 : {distance_from_detector_2}")

def add_detector(ax: Axes, pos: tuple[float, float], color: str, E_amp) -> None:
    """Add a detector to the plot."""
    # Show the position of the detectors
    detector_pos = Circle(
        (round(pos[0] / DELTA_X), round(pos[1] / DELTA_X)),
        M/100,
        color=color,
        fill=True,
    )
    ax.add_patch(detector_pos)

    # Show the distance from the source to the detector
    distance = distance_from_source(E_amp[round(pos[1]/DELTA_X), round(pos[0]/DELTA_X)])
    print(f"distance from source to detector : {distance}")
    circle = Circle(
        (round(pos[0] / DELTA_X), round(pos[1] / DELTA_X)),
        distance/DELTA_X,
        color=color,
        fill=False,
        linestyle="--",
        linewidth=2.5,
    )
    ax.add_patch(circle)

# circle_1 = Circle(
#     (round(DETECTOR_1_POS[0] / DELTA_X), round(DETECTOR_1_POS[1] / DELTA_X)),
#     distance_from_detector_1/DELTA_X,
#     color="red",
#     fill=False,
#     linestyle="--",
#     linewidth=3,
# )
# circle_2 = Circle(
#     (round(DETECTOR_2_POS[0] / DELTA_X), round(DETECTOR_2_POS[1] / DELTA_X)),
#     distance_from_detector_2/DELTA_X,
#     color="blue",
#     fill=False,
#     linestyle="--",
#     linewidth=3,
# )

# ax.add_patch(circle_1)
# ax.add_patch(circle_2)

# # Show the position of the detectors
# dectector_1_pos = Circle(
#     (round(DETECTOR_1_POS[0] / DELTA_X), round(DETECTOR_1_POS[1] / DELTA_X)),
#     M/100,
#     color="red",
#     fill=True,
# )
# dectector_2_pos = Circle(
#     (round(DETECTOR_2_POS[0] / DELTA_X), round(DETECTOR_2_POS[1] / DELTA_X)),
#     M/100,
#     color="blue",
#     fill=True,
# )
# ax.add_patch(dectector_1_pos)
# ax.add_patch(dectector_2_pos)

# add_detector(ax, DETECTOR_1_POS, "red", E_amp)
# add_detector(ax, DETECTOR_2_POS, "blue", E_amp)
# add_detector(ax, DETECTOR_3_POS, "green", E_amp)

plt.show()