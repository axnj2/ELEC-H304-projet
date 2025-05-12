import numpy as np

from yee_FDTD_2D import simulate_and_animate, e0, C_VIDE, simulate_and_plot
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

import xxhash

# parameters
# settings parameters
microwave_side_length = 0.357  # in meters
FREQ_REF = 1.8e9  # Hz
Q = 10000  # number of time samples
TOTAL_CURRENT = 0.01  # A
LASAGNA_WITH = 0.15  # in meters
LASAGNA_LENGTH = 0.2  # in meters

# derived parameters
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 1000  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
M = int(microwave_side_length / DELTA_X)  # number of space samples per dimension
print(f"M : {M}, total number of points : {M**2}")


# add the lasagna in the middle of the grid
# using the average of the relative permittivity of the ingredients
# (imagine that the lasagna was mixed before being put in the microwave)
lasagna_relative_permittivity = create_square(
    (
        microwave_side_length / 2 - LASAGNA_LENGTH / 2,
        microwave_side_length / 2 - LASAGNA_WITH / 2,
    ),
    LASAGNA_LENGTH,
    LASAGNA_WITH,
    (M, M),
    DELTA_X,
    value=float(np.mean(
        (
            meat_relative_real_permittivity,
            cheese_relative_real_permittivity,
            pasta_relative_real_permittivity,
            sauce_relative_real_permittivity,
        )
    )),
    default_value=1.0,
)

lasagna_conductivity = create_square(
    (
        microwave_side_length / 2 - LASAGNA_LENGTH / 2,
        microwave_side_length / 2 - LASAGNA_WITH / 2,
    ),
    LASAGNA_LENGTH,
    LASAGNA_WITH,
    (M, M),
    DELTA_X,
    value=float(np.mean(
        (
            meat_relative_complex_permittivity,
            cheese_relative_complex_permittivity,
            pasta_relative_complex_permittivity,
            sauce_relative_complex_permittivity,
        )
    )) * (2* np.pi * FREQ_REF * e0),
    default_value=0.0,
)

# create the source
hasher = xxhash.xxh64()

hasher.update(str(TOTAL_CURRENT).encode())
hasher.update(str(FREQ_REF).encode())

source_hash = hasher.hexdigest()

def source(q, J):
    sinusoïdal_point_source(
        J,
        q,
        M,
        int(0.320/DELTA_X),
        M//2,
        TOTAL_CURRENT,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
    )


# initialize the arrays
E0 = np.zeros((M, M), dtype=np.float32)
B_tilde_0 = np.zeros((M, M), dtype=np.float32)
J = np.zeros((M, M), dtype=np.float32)

# simulate_and_animate(
#     E0,
#     B_tilde_0,
#     DELTA_T,
#     DELTA_X,
#     1e-5,
#     all_time_max,
#     Q,
#     M,
#     source,
#     norm_type="lin",
#     use_progress_bar=True,
#     local_conductivity=lasagna_conductivity,
#     local_rel_permittivity=lasagna_relative_permittivity,
# )

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
im, E = simulate_and_plot(
    ax,
    DELTA_T,
    DELTA_X,
    Q,
    M,
    source,
    local_conductivity=lasagna_conductivity,
    local_rel_permittivity=lasagna_relative_permittivity,
    current_func_hash=source_hash,
)
plt.show()

    