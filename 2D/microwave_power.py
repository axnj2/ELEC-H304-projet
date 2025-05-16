import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from define_lasagna_microwave import initialize_mixed_lasagna
from yee_FDTD_2D import (
    compute_electric_field_amplitude,
    plot_field,
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


