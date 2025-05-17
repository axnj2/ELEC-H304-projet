import numpy as np

from yee_FDTD_2D import (
    compute_electric_field_amplitude_and_plot,
    get_exponential_decay_alpha,
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from define_lasagna_microwave import initialize_mixed_lasagna



# parameters
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
SLICE_POSITION = microwave_side_length / 2 + LASAGNA_WITH / 4


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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1: Axes = ax1
ax2: Axes = ax2

im, E_amplitude = compute_electric_field_amplitude_and_plot(
    ax1,
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1 / FREQ_REF,
    source,
    local_conductivity=local_conductivity,
    local_rel_permittivity=local_relative_permittivity,
    min_color_value=1e-1,
    show_material=True,
)


SLICE_INDEX = round(SLICE_POSITION / DELTA_X)
ax2.plot(
    np.linspace(0, LASAGNA_LENGTH, round(LASAGNA_LENGTH / DELTA_X)) * 1000,
    E_amplitude[
        round((microwave_side_length / 2 - LASAGNA_LENGTH / 2) / DELTA_X) : round(
            (microwave_side_length / 2 + LASAGNA_LENGTH / 2) / DELTA_X
        ),
        SLICE_INDEX,
    ],
)

ax2.set_xlabel("Distance [mm]")
ax2.set_ylabel("Amplitude du champs électrique [V/m]")
ax2.set_title("Amplitude du champs électrique dans la lasagne")
ax1.set_title("Amplitude du champs électrique dans le micro-onde")

alpha = get_exponential_decay_alpha(
    mixed_lasagna_conductivity,
    mixed_lasagna_rel_permittivity,
    1,
    2 * np.pi * FREQ_REF,
)
print(f"profondeur de peau : {1 / alpha:.2e} m")
ax2.axvline(
    1 / alpha * 1000,
    color="red",
    linestyle="--",
    label="λ théorique",
)
ax2.axvline(
    LASAGNA_LENGTH * 1000 - 1 / alpha * 1000,
    color="red",
    linestyle="--",
)
ax2.set_ylim(0, None)
ax2.set_xlim(0, LASAGNA_LENGTH * 1000)

initial_value_in_lasagna = np.max(
    E_amplitude[
        round((microwave_side_length / 2 - LASAGNA_LENGTH / 2) / DELTA_X) : round(
            (microwave_side_length / 2 + LASAGNA_LENGTH / 2) / DELTA_X
        ),
        SLICE_INDEX,
    ]
)

ax2.axhline(
    initial_value_in_lasagna / np.e, color="green", linestyle="--", label="A0/e"
)

ax2.legend()
plt.tight_layout()
plt.savefig("images/microwave_lasagna_field_magnitude.png", bbox_inches="tight", dpi=300)
# plt.show()
