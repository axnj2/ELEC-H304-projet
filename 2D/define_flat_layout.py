import numpy as np

from yee_FDTD_2D import C_VIDE, xp, using_cupy, TYPE_CHECKING
from simu_elements import (
    sinusoïdal_point_source,
    create_square_boundery,
    create_square,
    brick_wall_conductivity,
    brick_wall_rel_permittivity,
    wooden_door_conductivity,
    wooden_door_rel_permittivity,
)

import matplotlib.pyplot as plt

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
M = round(TOTAL_X / DELTA_X)  # number of space samples per dimension
print(f"M : {M}, total number of points : {M**2}")

all_walls = xp.zeros((M, M), dtype=np.bool)

center = np.array((TOTAL_X / 2 - 5, TOTAL_X / 2 -2))
# create the appartement layout
# using xor to remove parts of the walls
# my room = room1
room1_walls = (
    create_square_boundery(
        center, 4, 4, (M, M), DELTA_X, value=True, default_value=False, thickness=0.2
    )
    ^ create_square(
        center + np.array((0, 1)),
        0.2,
        2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )  # Window
    ^ create_square(
        center + np.array((0.5, 4)),
        0.8,
        -0.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,  # small_door
    )
    ^ create_square(
        center + np.array((4, 1)),
        -0.2,
        2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,  # big door
    )
)
# living room = room2
room2_walls = (
    create_square_boundery(
        center + np.array((0, 0.2)),
        10,
        -4.4,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    ^ create_square(  # Window front
        center + np.array((0, -0.7)),
        0.2,
        -3,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )  # front window
    ^ create_square(  # Window front
        center + np.array((10, -0.7)),
        -0.2,
        -3,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )  # front window
    ^ create_square(  # middle door
        center + np.array((4, 0.2)),
        0.8,
        -0.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    ^ create_square(  # middle door
        center + np.array((5.1, 0.2)),
        0.8,
        -0.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    | (
        create_square(  # middle wall
            center + np.array((4.9, 0.2)),
            0.2,
            -4.4,
            (M, M),
            DELTA_X,
            value=True,
            default_value=False,
        )
        ^ create_square(  # middle hole
            center + np.array((4.9, -0.5)),
            0.2,
            -3,
            (M, M),
            DELTA_X,
            value=True,
            default_value=False,
        )
    )
)
kitchen_walls = (
    create_square_boundery(
        center + np.array((6, 0)),
        4,
        4,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    ^ create_square(  # Window back
        center + np.array((10, 1.2)),
        -0.2,
        2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    ^ create_square(  # door
        center + np.array((6, 1.2)),
        0.2,
        2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
)
master_bedroom_walls = (
    create_square_boundery(
        center + np.array((0, 3.8)),
        4,
        4.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    ^ create_square(  # Window front
        center + np.array((0, 4.8)),
        0.2,
        2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    ^ create_square(  # door
        center + np.array((3.8, 4)),
        0.2,
        0.8,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
)

bathroom_walls = (
    create_square_boundery(
        center + np.array((7.5, 4.8)),
        2.5,
        3.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    ^ create_square(  # lower dorr
        center + np.array((7.8, 4.8)),
        0.8,
        0.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    ^ create_square(  # back window
        center + np.array((10, 5.5)),
        -0.2,
        1,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
)

restroom_wall = create_square(
    center + np.array((10, 4.8)),
    -0.2,
    -1,
    (M, M),
    DELTA_X,
    value=True,
    default_value=False,
)  # middle wall

stair_well_walls = create_square_boundery(
    center + np.array((3.8, 4.8)),
    3.9,
    3.2,
    (M, M),
    DELTA_X,
    value=True,
    default_value=False,
    thickness=0.2,
)


all_walls = (
    room2_walls
    | room1_walls
    | kitchen_walls
    | master_bedroom_walls
    | bathroom_walls
    | restroom_wall
    | stair_well_walls
)

metal_door = create_square(
    center + np.array((4, 4.8)),
    1.8,
    0.2,
    (M, M),
    DELTA_X,
    value=True,
    default_value=False,
)
everything = all_walls.astype(np.int16) + metal_door.astype(np.int16)

local_relative_permittivity = all_walls.astype(np.float32) * brick_wall_rel_permittivity
local_conductivity = all_walls.astype(np.float32) * brick_wall_conductivity
perfect_conductor_mask = metal_door

if __name__ == "__main__":
    # plot the walls
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("walls")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, TOTAL_X)
    ax.set_ylim(0, TOTAL_X)
    ax.set_aspect("equal")
    if using_cupy and not TYPE_CHECKING:
        everything = xp.asnumpy(everything)
    im = plt.imshow(
        everything,
        origin="lower",
        cmap="gray",
        extent=(0, TOTAL_X, 0, TOTAL_X),
    )
    plt.show()
