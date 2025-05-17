import numpy as np
import math

from yee_FDTD_2D import u0, e0, C_VIDE, xp, using_cupy, TYPE_CHECKING
from simu_elements import (
    sinusoïdal_point_source,
    create_square_boundery,
    create_square,
    brick_wall_loss_tangent,
    brick_wall_rel_permittivity,

)

import matplotlib.pyplot as plt

# parameters
TOTAL_X = 50  # in meters
FREQ_REF = 2.4e9  # Hz
# max time step : 6000 with 20 samples per wavelength and 25 m total size
Q = 12000  # number of time samples
ceiling_height = 2.5  # in meters
mesured_power = 1.8330e+1 # in W/m for I = 0.01 A
POWER_PROP_CONSTANT = mesured_power / (0.01**2)  # W/mA^2
POWER = 0.1 # in W
TOTAL_CURRENT = math.sqrt(POWER / (POWER_PROP_CONSTANT * ceiling_height))  # A
CENTER = np.array((TOTAL_X / 2 - 5, TOTAL_X / 2 - 2))
SOURCE_POSITION = np.array((TOTAL_X/2, TOTAL_X/2))  # in meters

# derived parameters
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 20  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
M = round(TOTAL_X / DELTA_X)  # number of space samples per dimension
print(f"M : {M}, total number of points : {M**2}")
speed_of_light_in_wall = np.sqrt(1/(u0*e0*brick_wall_rel_permittivity))
print(f"speed of light in wall : {speed_of_light_in_wall:.2e}")
print(f"number of time steps per wavelength in wall : {speed_of_light_in_wall/(DELTA_X * FREQ_REF):.2f}")

all_walls = xp.zeros((M, M), dtype=np.bool)

# create the appartement layout
# using xor to remove parts of the walls
# my room = room1
room1_walls = (
    create_square_boundery(
        CENTER, 4, 4, (M, M), DELTA_X, value=True, default_value=False, thickness=0.2
    )
    ^ create_square(
        CENTER + np.array((0, 1)),
        0.2,
        2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )  # Window
    ^ create_square(
        CENTER + np.array((0.5, 4)),
        0.8,
        -0.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,  # small_door
    )
    ^ create_square(
        CENTER + np.array((4, 1)),
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
        CENTER + np.array((0, 0.2)),
        10,
        -4.4,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    ^ create_square(  # Window front
        CENTER + np.array((0, -0.7)),
        0.2,
        -3,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )  # front window
    ^ create_square(  # Window front
        CENTER + np.array((10, -0.7)),
        -0.2,
        -3,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )  # front window
    ^ create_square(  # middle door
        CENTER + np.array((4, 0.2)),
        0.8,
        -0.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    ^ create_square(  # middle door
        CENTER + np.array((5.1, 0.2)),
        0.8,
        -0.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    | (
        create_square(  # middle wall
            CENTER + np.array((4.9, 0.2)),
            0.2,
            -4.4,
            (M, M),
            DELTA_X,
            value=True,
            default_value=False,
        )
        ^ create_square(  # middle hole
            CENTER + np.array((4.9, -0.5)),
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
        CENTER + np.array((6, 0)),
        4,
        4,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    ^ create_square(  # Window back
        CENTER + np.array((10, 1.2)),
        -0.2,
        2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    ^ create_square(  # door
        CENTER + np.array((6, 1.2)),
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
        CENTER + np.array((0, 3.8)),
        4,
        4.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    ^ create_square(  # Window front
        CENTER + np.array((0, 4.8)),
        0.2,
        2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    ^ create_square(  # door
        CENTER + np.array((3.8, 4)),
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
        CENTER + np.array((7.5, 4.8)),
        2.5,
        3.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
        thickness=0.2,
    )
    ^ create_square(  # lower dorr
        CENTER + np.array((7.8, 4.8)),
        0.8,
        0.2,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
    ^ create_square(  # back window
        CENTER + np.array((10, 5.5)),
        -0.2,
        1,
        (M, M),
        DELTA_X,
        value=True,
        default_value=False,
    )
)

restroom_wall = create_square(
    CENTER + np.array((10, 4.8)),
    -0.2,
    -1,
    (M, M),
    DELTA_X,
    value=True,
    default_value=False,
)  # middle wall

stair_well_walls = create_square_boundery(
    CENTER + np.array((3.8, 4.8)),
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
    CENTER + np.array((4, 4.8)),
    1.8,
    0.2,
    (M, M),
    DELTA_X,
    value=True,
    default_value=False,
)
everything = all_walls.astype(np.int16) + metal_door.astype(np.int16)
everything[np.floor(SOURCE_POSITION/DELTA_X).astype(int)[1],np.floor(SOURCE_POSITION/DELTA_X).astype(int)[0]] = 10  # source

local_relative_permittivity = xp.ones((M, M), dtype=np.float32)
local_relative_permittivity[all_walls] = brick_wall_rel_permittivity
brick_wall_conductivity = xp.zeros((M, M), dtype=np.float32)
brick_wall_conductivity[all_walls] = (
    brick_wall_loss_tangent * 2 * np.pi * FREQ_REF * brick_wall_rel_permittivity * e0
)
local_conductivity = all_walls.astype(np.float32) * brick_wall_conductivity
perfect_conductor_mask = metal_door

def source(q, J):
    sinusoïdal_point_source(
        J,
        q,
        M,
        round(SOURCE_POSITION[0] / DELTA_X),
        round(SOURCE_POSITION[1] / DELTA_X),
        TOTAL_CURRENT,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
    )


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
