import numpy as np
import math

from yee_FDTD_2D import (
    e0,
    u0,
    C_VIDE,
    using_cupy,
    TYPE_CHECKING,
    xp,
)
from simu_elements import (
    sinusoïdal_point_source,
    create_square,
    meat_relative_real_permittivity,
    meat_relative_complex_permittivity,
    pasta_relative_complex_permittivity,
    pasta_relative_real_permittivity,
    cheese_relative_complex_permittivity,
    cheese_relative_real_permittivity,
    sauce_relative_complex_permittivity,
    sauce_relative_real_permittivity,
)

import matplotlib.pyplot as plt


# parameters
# settings parameters
microwave_side_length = 0.357  # in meters
FREQ_REF = 1.8e9  # Hz
Q = 15000  # number of time samples
POWER_PROP_CONSTANT = 2273417.16532613 # W/mA^2
POWER = 1000 # W
LASAGNA_HEIGHT = 0.1 # in meters
TOTAL_CURRENT = math.sqrt(POWER / (POWER_PROP_CONSTANT * LASAGNA_HEIGHT))  # in Amperes
LASAGNA_WITH = 0.15  # in meters
LASAGNA_LENGTH = 0.2  # in meters

# derived parameters
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 400  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
M = int(microwave_side_length / DELTA_X)  # number of space samples per dimension
print(f"M : {M}, total number of points : {M**2}")

mixed_lasagna_rel_permittivity = float(
    np.mean(
        (
            meat_relative_real_permittivity,
            cheese_relative_real_permittivity,
            pasta_relative_real_permittivity,
            sauce_relative_real_permittivity,
        )
    )
)
mixed_lasagna_conductivity = float(
    np.mean(
        (
            meat_relative_complex_permittivity,
            cheese_relative_complex_permittivity,
            pasta_relative_complex_permittivity,
            sauce_relative_complex_permittivity,
        )
    )
    * (2 * np.pi * FREQ_REF * e0)
)
speed_of_light_in_lasagna = 1 / np.sqrt(mixed_lasagna_rel_permittivity * e0 * u0)


source_position = (
    0.320,
    microwave_side_length / 2,
)


def source(q, J):
    sinusoïdal_point_source(
        J,
        q,
        M,
        round(source_position[0] / DELTA_X),
        round(source_position[1] / DELTA_X),
        TOTAL_CURRENT,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
    )


def initialize_mixed_lasagna():
    """Initialize the lasagna parameters."""
    # add the lasagna in the middle of the grid
    # using the average of the relative permittivity of the ingredients
    # (imagine that the lasagna was mixed before being put in the microwave)
    local_relative_permittivity = create_square(
        (
            microwave_side_length / 2 - LASAGNA_LENGTH / 2,
            microwave_side_length / 2 - LASAGNA_WITH / 2,
        ),
        LASAGNA_LENGTH,
        LASAGNA_WITH,
        (M, M),
        DELTA_X,
        value=mixed_lasagna_rel_permittivity,
        default_value=1.0,
    )

    local_conductivity = create_square(
        (
            microwave_side_length / 2 - LASAGNA_LENGTH / 2,
            microwave_side_length / 2 - LASAGNA_WITH / 2,
        ),
        LASAGNA_LENGTH,
        LASAGNA_WITH,
        (M, M),
        DELTA_X,
        value=mixed_lasagna_conductivity,
        default_value=0.0,
    )
    return (
        DELTA_T,
        DELTA_X,
        Q,
        M,
        FREQ_REF,
        local_relative_permittivity,
        local_conductivity,
        source,
        LASAGNA_LENGTH,
        LASAGNA_WITH,
        microwave_side_length,
        mixed_lasagna_rel_permittivity,
        mixed_lasagna_conductivity,
    )


if __name__ == "__main__":
    (
        DELTA_T,
        DELTA_X,
        Q,
        M,
        FREQ_REF,
        local_relative_permittivity,
        local_conductivity,
        source,
        LASAGNA_LENGTH,
        LASAGNA_WITH,
        microwave_side_length,
        mixed_lasagna_rel_permittivity,
        mixed_lasagna_conductivity,
    ) = initialize_mixed_lasagna()
    print(f"total current : {TOTAL_CURRENT:.2e} A")
    print(f"speed_of_light_in_lasagna : {speed_of_light_in_lasagna:.2e} m/s")
    print(f"sample per wavelength : {speed_of_light_in_lasagna / (DELTA_X * FREQ_REF)}")
    print(f"lasagna conductivity : {mixed_lasagna_conductivity:.2e} S/m")
    print(f"thermal inertia : {(1010.25 * 3670):.2e} J/Km^3")
    print(f"wave period : {1/FREQ_REF:.2e} s")
    print(f"wave length in vacum : {WAVE_LENGTH:.2e} m")
    # plot in black and white
    plt.figure(figsize=(8, 8))
    plt.title("Local relative permittivity of the lasagna")

    if using_cupy and not TYPE_CHECKING:
        local_relative_permittivity = xp.asnumpy(local_relative_permittivity)
    plt.imshow(
        local_relative_permittivity,
        cmap="gray",
        origin="lower",
    )
    plt.show()
