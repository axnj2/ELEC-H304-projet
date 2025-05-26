import numpy as np

from yee_FDTD_2D import (
    compute_electric_field_amplitude,
    plot_field,
    field_to_power,
    e0,
    C_VIDE,
    compute_mean_poynting_integral,
    create_material_image,
)
from simu_elements import sinusoïdal_point_source, create_square_boundery

from matplotlib import pyplot as plt

# parameters
# settings parameters
TOTAL_X = 300  # in meters
FREQ_REF = 1e8  # Hz
Q = 2500  # number of time samples
TOTAL_CURRENT_SINGLE = 0.0402  # A
TOTAL_CURRENT_ARRAY = 0.095  # A
MIN_COLOR = 1e-3  # minimum color value for the image
ANTENNA_RESISTANCE = 50  # [Ohm]
ANTENNA_EQUIVALENT_HEIGHT = 1  # [m]
ROI_WIDTH = 135  # [m]
ROI_HEIGHT = 180  # [m]

# derived parameters
WAVE_LENGTH = C_VIDE / FREQ_REF  # in meters
REFINEMENT_FACTOR = 20  # number of samples per wavelength (min 20)
DELTA_X = WAVE_LENGTH / REFINEMENT_FACTOR  # in meters
DELTA_T = 1 / (2 * FREQ_REF * REFINEMENT_FACTOR)  # in seconds
M = round(TOTAL_X / DELTA_X)  # number of space samples per dimension
all_time_max = TOTAL_CURRENT_SINGLE / (DELTA_X * DELTA_X) * DELTA_T / e0

print(f"M : {M}, total number of points : {M**2}")

# Phase array parameters
# simple source : https://www.youtube.com/watch?v=jSDLfcNhThw
NUMBER_OF_ELEMENTS = 10  # number of elements in the phased array
ELEMENT_SPACING = WAVE_LENGTH / 2  # [m]
ELEMENT_SPACING_INDEX = int(
    ELEMENT_SPACING / DELTA_X
)  # number of grid points between elements
TARGET_ANGLE = 180  # [degrees]


def phase_distribution(index: int) -> float:
    return (
        np.pi
        * index
        * ELEMENT_SPACING
        * np.sin(TARGET_ANGLE * np.pi / 180)
        / WAVE_LENGTH
    )


def current_func_phased_array(q: int, previous_J):
    start_index = int(M / 2) - ELEMENT_SPACING_INDEX * (NUMBER_OF_ELEMENTS - 1) // 2
    for ii in range(NUMBER_OF_ELEMENTS):
        sinusoïdal_point_source(
            previous_J,
            q,
            M,
            int(1 / DELTA_X),
            start_index + ii * ELEMENT_SPACING_INDEX,
            TOTAL_CURRENT_ARRAY
            / np.sqrt(
                NUMBER_OF_ELEMENTS
            ),  # power for a sinusoidal source is prop to I^2 => divide by sqrt(N)
            FREQ_REF,
            DELTA_T,
            DELTA_X,
            phase=phase_distribution(ii),
        )


def current_func_sinusoidal(q: int, previous_J):
    sinusoïdal_point_source(
        previous_J,
        q,
        M,
        int(1 / DELTA_X),
        M // 2,
        TOTAL_CURRENT_SINGLE,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
    )


# phased_array emitted power by poynting
phased_array_emitted_power = compute_mean_poynting_integral(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1 / FREQ_REF,
    current_func_phased_array,
    (0, DELTA_X * (int(M / 2) - ELEMENT_SPACING_INDEX * (NUMBER_OF_ELEMENTS - 1) // 2)+1),
    (5, DELTA_X * (int(M / 2) + ELEMENT_SPACING_INDEX * (NUMBER_OF_ELEMENTS - 1) // 2)+1),
)
print(f"Phased array emitted power : {phased_array_emitted_power:.2e} W")
Power_phased_array = field_to_power(
    compute_electric_field_amplitude(
        DELTA_T,
        DELTA_X,
        Q,
        M,
        1 / FREQ_REF,
        current_func_phased_array,
    ),
    R_a=ANTENNA_RESISTANCE,
    h_e=ANTENNA_EQUIVALENT_HEIGHT,
)

roi_width_steps = int(ROI_WIDTH / DELTA_X)
roi_height_steps = int(ROI_HEIGHT / DELTA_X)
roi_x_start = 0
roi_x_end = roi_width_steps
roi_y_start = int(M / 2) - roi_height_steps // 2
roi_y_end = roi_y_start + roi_height_steps

Selected_Field_phased_array = Power_phased_array[
    roi_y_start:roi_y_end, roi_x_start:roi_x_end
]

# sinusoidal source emitted power by poynting
sinusoidal_emitted_power = compute_mean_poynting_integral(
    DELTA_T,
    DELTA_X,
    Q,
    M,
    1 / FREQ_REF,
    current_func_sinusoidal,
    (0, TOTAL_X / 2 - 10),
    (20, TOTAL_X / 2 + 10),
)
print(f"Sinusoidal source emitted power : {sinusoidal_emitted_power:.2e} W")

Power_sinusoidal = field_to_power(
    compute_electric_field_amplitude(
        DELTA_T,
        DELTA_X,
        Q,
        M,
        1 / FREQ_REF,
        current_func_sinusoidal,
    ),
    R_a=ANTENNA_RESISTANCE,
    h_e=ANTENNA_EQUIVALENT_HEIGHT,
)

Selected_Field_sinusoidal = Power_sinusoidal[
    roi_y_start:roi_y_end, roi_x_start:roi_x_end
]

# maximum value for all the images
max_value = max(
    np.max(Selected_Field_phased_array),
    np.max(Selected_Field_sinusoidal),
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


im = plot_field(
    ax1,
    DELTA_X,
    Selected_Field_phased_array,
    norm_type="log",
    min_color_value=MIN_COLOR,
    max_color_value=max_value,
    color_bar=False,
    
)

ax1.set_title("Phased array power distribution")

# get the existing colorbar
colorbar = ax1.images[-1].colorbar
if colorbar is not None:
    colorbar.set_label("Power [W]")

im = plot_field(
    ax2,
    DELTA_X,
    Selected_Field_sinusoidal,
    norm_type="log",
    min_color_value=MIN_COLOR,
    max_color_value=max_value,
    # image_overlay=create_material_image(
    #     None,
    #     None,
    #     create_square_boundery(
    #         (0, TOTAL_X / 2 - 5),
    #         10,
    #         10,
    #         (M, M),
    #         DELTA_X,
    #         True,
    #         False,
    #         thickness=1,
    #     )[roi_y_start:roi_y_end, roi_x_start:roi_x_end],
    #     n_max=roi_x_end - roi_x_start,
    #     m_max=roi_y_end - roi_y_start,
    #     show_material=True,
    #     show_edges_of_materials=False,
    # ),
)

ax2.set_title("Sinusoidal source power distribution")
# get the existing colorbar
colorbar = ax2.images[-1].colorbar
if colorbar is not None:
    colorbar.set_label("Power [W]")

plt.savefig(
    "images/phased_array_power_distribution_VS_sinuoidal.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
