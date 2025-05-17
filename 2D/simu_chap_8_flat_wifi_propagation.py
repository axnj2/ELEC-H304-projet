import numpy as np

from yee_FDTD_2D import (
    compute_mean_poynting_integral,
    simulate_and_animate,
    compute_electric_field_amplitude,
    plot_field,
    using_cupy,
    TYPE_CHECKING,
    xp,
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from define_flat_layout import (
    TOTAL_X,
    DELTA_T,
    DELTA_X,
    Q,
    M,
    FREQ_REF,
    local_relative_permittivity,
    local_conductivity,
    perfect_conductor_mask,
    source,
    CENTER,
    SOURCE_POSITION,
    ceiling_height,
)

# E0 = np.zeros((M, M), dtype=np.float32)
# B_tilde_0 = np.zeros((M, M), dtype=np.float32)
# J = np.zeros((M, M), dtype=np.float32)


# simulate_and_animate(
#     E0,
#     B_tilde_0,
#     DELTA_T,
#     DELTA_X,
#     1e-2,
#     10,
#     Q,
#     M,
#     source,
#     norm_type="lin",
#     use_progress_bar=True,
#     local_conductivity=local_conductivity,
#     local_rel_permittivity=local_relative_permittivity,
#     perfect_conductor_mask=perfect_conductor_mask,
#     show_edges_of_materials=False,
#     show_from=5000,
# )

fig, ax = plt.subplots(figsize=(10, 10))

E_amp = compute_electric_field_amplitude(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1/FREQ_REF,
    source,
    local_conductivity=local_conductivity,
    local_rel_permittivity=local_relative_permittivity,
    perfect_conductor_mask=perfect_conductor_mask,
)

power = compute_mean_poynting_integral(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1 / FREQ_REF,
    source,
    upper_left_corner=(SOURCE_POSITION[0] - 0.2, SOURCE_POSITION[1] - 0.2),
    lower_right_corner=(SOURCE_POSITION[0] + 0.2, SOURCE_POSITION[1] + 0.2),
    local_conductivity=local_conductivity,
    local_rel_permittivity=local_relative_permittivity,
    perfect_conductor_mask=perfect_conductor_mask,
)
print(f"power imited by the source: {power * ceiling_height:.4e} W/m")

flat_min_x = round((CENTER[0] ) / DELTA_X)
flat_max_x = round((CENTER[0] + 10) / DELTA_X)
flat_min_y = round((CENTER[1] - 4.2) / DELTA_X)
flat_max_y = round((CENTER[1] + 8) / DELTA_X)

E_amp = E_amp[flat_min_y:flat_max_y, flat_min_x:flat_max_x]

im = plot_field(
    ax,
    DELTA_X,
    E_amp,
    min_color_value=0.01,
)

plt.title("Amplitude du champ électrique")
plt.savefig("image/flat_wifi_propagation_amplitude.png", dpi=300, bbox_inches="tight")
plt.show()
