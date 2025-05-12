import numpy as np

from yee_FDTD_2D import simulate_and_animate, e0, C_VIDE, xp
from simu_elements import (
    sinusoïdal_point_source,
    create_square_boundery,
    create_square,
    brick_wall_conductivity,
    brick_wall_rel_permittivity,
    wooden_door_conductivity,
    wooden_door_rel_permittivity,
)

import pyqtgraph as pg

# parameters
TOTAL_X = 50  # in meters
FREQ_REF = 2.4e9  # Hz
Q = 1000  # number of time samples
TOTAL_CURRENT = 0.01  # A

# derived parameters
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 20  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
M = int(TOTAL_X / DELTA_X)  # number of space samples per dimension
print(f"M : {M}, total number of points : {M**2}")

all_walls = xp.zeros((M, M), dtype=np.bool)

center = np.array((TOTAL_X / 2, TOTAL_X / 2))
print(f"center : {center}, type : {type(center)}")
print(f"center + np.array((4, 0.5)) : {center + np.array((4, 0.5))}, type : {type(center + np.array((4, 0.5)))}")
# create the appartement layout
# my room = room1
# room1_walls = (
#     create_square_boundery(
#         center, 4, 4, (M, M), DELTA_X, value=True, default_value=False, thickness=0.2
#     )
#     ^ create_square(
#         center + np.array((1, 0)),
#         2,
#         0.2,
#         (M, M),
#         DELTA_X,
#         value=True,
#         default_value=False,
#     )  # Window
#     ^ create_square(
#         center + np.array((4, 0.5)),
#         -0.2,
#         0.8,
#         (M, M),
#         DELTA_X,
#         value=True,
#         default_value=False,  # small_door
#     )
#     ^ create_square(
#         center + np.array((1, 4)),
#         2,
#         -0.2,
#         (M, M),
#         DELTA_X,
#         value=True,
#         default_value=False,  # big door
#     )
# )
# living room = room2
room2_walls = (
    create_square_boundery(
        center + np.array((0.2, 0)),
        12,
        -5,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    # ^ create_square(  # Window front
    #     center + np.array((-4.5, 0)),
    #     0.2,
    #     4,
    #     (M, M),
    #     DELTA_X,
    #     value=True,
    #     default_value=False,
    # )
    # ^ create_square(  # window back
    #     center + np.array((-4.5, 12)),
    #     -0.2,
    #     4,
    #     (M, M),
    #     DELTA_X,
    #     value=True,
    #     default_value=False,
    # )
)
all_walls = room2_walls

plot = pg.plot()
plot.setFixedSize(800, 800)
im = pg.ImageItem(
    (all_walls).get(),
)
plot.addItem(im)
pg.exec()
