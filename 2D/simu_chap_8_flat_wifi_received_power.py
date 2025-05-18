import numpy as np
import math

from yee_FDTD_2D import (
    compute_mean_poynting_integral,
    simulate_and_animate,
    compute_electric_field_amplitude,
    plot_field,
    using_cupy,
    TYPE_CHECKING,
    xp,
    field_to_power,
    C_VIDE,
    power_to_dBm,
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
ANTENNA_RESISTANCE = 50 # Ohm
PEAK_GAIN = 10**(1.5 / 10)  # dBi
FREE_SPACE_IMPEDANCE = 377  # Ohm
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
EQUIVALENT_HEIGHT = WAVE_LENGTH * math.sqrt(PEAK_GAIN*ANTENNA_RESISTANCE / (math.pi * FREE_SPACE_IMPEDANCE)) # in meters
print(f"Equivalent height of the antenna : {EQUIVALENT_HEIGHT:.2e} m")

fig, ax = plt.subplots(figsize=(10, 10))

E_amp = compute_electric_field_amplitude(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1 / FREQ_REF,
    source,
    local_conductivity=local_conductivity,
    local_rel_permittivity=local_relative_permittivity,
    perfect_conductor_mask=perfect_conductor_mask,
)

flat_min_x = round((CENTER[0]) / DELTA_X)
flat_max_x = round((CENTER[0] + 10) / DELTA_X)
flat_min_y = round((CENTER[1] - 4.2) / DELTA_X)
flat_max_y = round((CENTER[1] + 8) / DELTA_X)

E_amp = E_amp[flat_min_y:flat_max_y, flat_min_x:flat_max_x]

im = plot_field(
    ax,
    DELTA_X,
    power_to_dBm(field_to_power(E_amp, ANTENNA_RESISTANCE, EQUIVALENT_HEIGHT)),
    min_color_value=-80,
    norm_type="asymlin",
    color_bar_label="Puissance reçue (dBm)",
)

plt.title("Puissance reçue par un appareil wifi")
plt.savefig("images/flat_wifi_propagation_puissance_reçue.png", dpi=300, bbox_inches="tight")
plt.show()
