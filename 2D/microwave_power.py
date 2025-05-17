import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from define_lasagna_microwave import initialize_mixed_lasagna, source_position, LASAGNA_HEIGHT
from simu_elements import create_square_boundery
from yee_FDTD_2D import (
    compute_mean_poynting_integral,
)

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

# show integration path and lasagna
fig, ax = plt.subplots(figsize=(10, 10))
ax: Axes = ax
integration_path = create_square_boundery(
    (source_position[0] - 0.01, source_position[1] - 0.01),
    0.02,
    0.02,
    (M, M),
    DELTA_X,
    True,
    False,
    DELTA_X,
)
image = integration_path  + local_conductivity
ax.imshow(image.get())


power_per_distance_unit = compute_mean_poynting_integral(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1 / FREQ_REF,
    source,
    upper_left_corner=(source_position[0] - 0.01, source_position[1] - 0.01),
    lower_right_corner=(source_position[0] + 0.01, source_position[1] + 0.01),
    local_conductivity=local_conductivity,
    local_rel_permittivity=local_relative_permittivity,
)



print(f"Power absorbed in the lasagna: {power_per_distance_unit * LASAGNA_HEIGHT:.4e} W")
plt.show()