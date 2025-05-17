from typing import TYPE_CHECKING, Tuple


import numpy as xp
import numpy as np

if TYPE_CHECKING:
    using_cupy = False
    pass
else:
    using_cupy = False
    try:
        import cupy

        using_cupy = False
        if cupy.cuda.is_available():
            xp = cupy
            print("Using cupy")
            using_cupy = True
        else:
            print("Using numpy")
    except ImportError:
        print("Using numpy")


import math

from typing import (
    Callable,
)  # https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints

import pyqtgraph as pg

import copy

from tqdm import tqdm

from pyqtgraph.Qt import QtCore
import pyqtgraph.exporters

from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from scipy.ndimage import convolve


import os
import xxhash


# Constants
e0: float = 8.8541878188e-12  # F/m
u0: float = 1.25663706127e-6  # H/m
C_VIDE: float = 1 / math.sqrt(e0 * u0)  # m/s

# material color
MATERIAL_COLOR_FULL = (87, 87, 87, 100)  # purple
MATERIAL_COLOR_EDGE = (0, 255, 0, 255)  # red


def forward_E(
    E: xp.ndarray,
    B_tilde_x: xp.ndarray,
    B_tilde_y: xp.ndarray,
    q: int,
    M: int,
    dt: float,
    dx: float,
    J: xp.ndarray,
    epsilon_r: xp.ndarray | None,
    local_conductivity: xp.ndarray | None,
):
    """Performs a time step forward for the electric field E.
    Assumes a square grid

    Args:
        E (xp.ndarray): in [V/m] 2D array of the values of the electric field in the z direction on the main grid
        B_tilde_x (xp.ndarray): in [T] 2D array of the values of the magnetic field in the x direction on a grid shifted by 1/2 in the y direction
        B_tilde_y (xp.ndarray): in [T] 2D array of the values of the magnetic field in the y direction on a grid shifted by 1/2 in the x direction
        dt (float, optional): time step in [s]. Defaults to DELTA_T.
        dx (float, optional): space step in [m]. Defaults to DELTA_X.
        q (int): time step number
        current_source_func (Callable[[int], xp.ndarray], optional): function to get the current density in [A/m^2]. Defaults to get_source_J.
        epsilon_r (xp.ndarray | None, optional): map of the local relative permittivity value. Defaults to None.
    """

    # set the epsilon_r to 1 if not provided
    if epsilon_r is None:
        epsilon_r = xp.ones((M, M), dtype=xp.float32)

    # set the local conductivity to 0 if not provided
    if local_conductivity is None:
        local_conductivity = xp.zeros((M, M), dtype=xp.float32)

    # update the electric field
    E[1:M, 1:M] = (
        1 / (1 + local_conductivity[1:M, 1:M] * dt / (epsilon_r[1:M, 1:M] * e0 * 2))
    ) * (
        E[1:M, 1:M]
        * (1 - local_conductivity[1:M, 1:M] * dt / (epsilon_r[1:M, 1:M] * e0 * 2))
        + dt
        / (C_VIDE * e0 * epsilon_r[1:M, 1:M] * u0 * dx)
        * (
            -(B_tilde_x[1:M, 1:M] - B_tilde_x[0 : M - 1, 1:M])
            + (B_tilde_y[1:M, 1:M] - B_tilde_y[1:M, 0 : M - 1])
        )
        - dt / (e0 * epsilon_r[1:M, 1:M]) * J[1:M, 1:M]
    )

    # set the boundary conditions
    E[-1, :] = xp.ones((M), dtype=xp.float32) * 1e-30
    E[:, -1] = xp.ones((M), dtype=xp.float32) * 1e-30
    E[0, :] = xp.ones((M), dtype=xp.float32) * 1e-30
    E[:, 0] = xp.ones((M), dtype=xp.float32) * 1e-30


def forward_B_tilde(
    E: xp.ndarray,
    B_tilde_x: xp.ndarray,
    B_tilde_y: xp.ndarray,
    M: int,
    dt: float,
    dx: float,
):
    """Performs a time step forward for the magnetic field B_tilde.
    Assumes a square grid

    Args:
        E (xp.ndarray): in [V/m] 2D array of the values of the electric field in the z direction on the main grid
        B_tilde_x (xp.ndarray): in [T] 2D array of the values of the magnetic field in the x direction on a grid shifted by 1/2 in the y direction
        B_tilde_y (xp.ndarray): in [T] 2D array of the values of the magnetic field in the y direction on a grid shifted by 1/2 in the x direction
        dt (float, optional): time step in [s]. Defaults to DELTA_T.
        dx (float, optional): space step in [m]. Defaults to DELTA_X.
    """

    # update the magnetic field
    B_tilde_x[0 : M - 1, 0 : M - 1] = B_tilde_x[
        0 : M - 1, 0 : M - 1
    ] - C_VIDE * dt / dx * (E[1:M, 0 : M - 1] - E[0 : M - 1, 0 : M - 1])
    #                          \y index  |
    #                                     \x index
    B_tilde_y[0 : M - 1, 0 : M - 1] = B_tilde_y[
        0 : M - 1, 0 : M - 1
    ] + C_VIDE * dt / dx * (E[0 : M - 1, 1:M] - E[0 : M - 1, 0 : M - 1])


def step_yee(
    E: xp.ndarray,
    B_tilde_x: xp.ndarray,
    B_tilde_y: xp.ndarray,
    J: xp.ndarray,
    q: int,
    dt: float,
    dx: float,
    epsilon_r: xp.ndarray | None,
    current_source_func: Callable[[int, xp.ndarray], None] | None,
    perferct_conductor_mask: xp.ndarray | None,
    local_conductivity: xp.ndarray | None,
):
    """Performs a time step for the Yee algorithm.
    This function updates the electric field E and the magnetic field B_tilde_x
    and B_tilde_y
    using the Yee algorithm.

    Will set the electric field back to zero in the perfect conductor region.

    It assumes a square grid.

    Args:
        E (xp.ndarray): in [V/m] 2D array of the values of the electric field in the z direction on the main grid
        B_tilde_x (xp.ndarray): in [T] 2D array of the values of the magnetic field in the x direction on a grid shifted by 1/2 in the y direction
        B_tilde_y (xp.ndarray): in [T] 2D array of the values of the magnetic field in the y direction on a grid shifted by 1/2 in the x direction
        q (int): time step number
        dt (float, optional): time step in [s]. Defaults to DELTA_T.
        dx (float, optional): space step in [m]. Defaults to DELTA_X.
        current_source_func (Callable[[int], xp.ndarray], optional): function to get the current density in [A/m^2]. Defaults to get_source_J.
        perferct_conductor_mask (xp.ndarray | None, optional): mask of the perfect conductor region. Defaults to None.
    """
    # infer the grid size
    M = E.shape[0]

    # get the current density
    if current_source_func is not None:
        current_source_func(q, J)

    # validate that the dimensions are coeherent
    assert E.shape[0] == E.shape[1], "Error: E must be a square matrix"
    assert E.shape == B_tilde_x.shape, "Error: E and B_tilde_x must have the same shape"
    assert E.shape == B_tilde_y.shape, "Error: E and B_tilde_y must have the same shape"
    if perferct_conductor_mask is not None:
        assert E.shape == perferct_conductor_mask.shape, (
            "Error: E and perfect conductor mask must have the same shape"
        )
    if current_source_func is not None:
        assert E.shape == J.shape, "Error: E and J must have the same shape"

    forward_B_tilde(E, B_tilde_x, B_tilde_y, M, dt, dx)
    forward_E(
        E,
        B_tilde_x,
        B_tilde_y,
        q,
        M,
        dt,
        dx,
        J,
        epsilon_r,
        local_conductivity,
    )

    if perferct_conductor_mask is not None:
        E[perferct_conductor_mask] = 1e-30


def get_material_mask(
    local_conductivity: xp.ndarray | None,
    local_rel_permittivity: xp.ndarray | None,
    perfect_conductor_mask: xp.ndarray | None,
) -> np.ndarray | None:
    # show the obstacles based on the local relative permittivity or conductivity
    # returns a mask indicating where the local relative permittivity is not 1 and or the local conductivity is not 0
    match [local_conductivity, local_rel_permittivity]:
        case [None, None]:
            mask = None
        case [None, _]:
            mask = local_rel_permittivity != 1.0
        case [_, None]:
            mask = local_conductivity != 0.0
        case _:
            mask = np.logical_or(
                local_conductivity != 0.0, local_rel_permittivity != 1.0
            )
    if perfect_conductor_mask is not None:
        if mask is None:
            mask = perfect_conductor_mask
        else:
            mask = np.logical_or(mask, perfect_conductor_mask)

    if using_cupy:
        mask = xp.asnumpy(mask)  # type: ignore

    return mask


def get_material_edges(
    local_conductivity, local_rel_permittivity, perfect_conductor_mask
) -> np.ndarray | None:
    # show the obstacles edges based on the local relative permittivity or conductivity

    mask = get_material_mask(
        local_conductivity, local_rel_permittivity, perfect_conductor_mask
    )
    return get_material_edges_from_mask(mask)


def get_material_edges_from_mask(mask: np.ndarray | None) -> np.ndarray | None:
    # show the obstacles edges based on the local relative permittivity or conductivity

    if mask is not None:
        # get the edges of the mask
        edges = np.zeros_like(mask, dtype=np.uint8)
        # use convolution to get the edges
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ],
            dtype=np.int8,
        )
        edges = convolve(mask.astype(np.int8), kernel, mode="constant", cval=0)
        edges = edges > 0

        return edges
    else:
        return None


def create_material_image(
    local_conductivity: np.ndarray | None,
    local_rel_permittivity: np.ndarray | None,
    perfect_conductor_mask: np.ndarray | None,
    m_max: int,
    show_material: bool = True,
    show_edges_of_materials: bool = True,
) -> np.ndarray | None:
    mask = get_material_mask(
        local_conductivity, local_rel_permittivity, perfect_conductor_mask
    )

    material_image = None
    if mask is not None and show_material:
        if show_edges_of_materials:
            mask = get_material_edges_from_mask(mask)
            mat_color = MATERIAL_COLOR_EDGE
        else:
            mat_color = MATERIAL_COLOR_FULL
        material_image = np.zeros((m_max, m_max, 4), dtype=np.uint8)
        material_image[mask, :] = np.asarray(mat_color)

    return material_image


def simulate_and_animate(
    E0: xp.ndarray,
    B_tilde_0: xp.ndarray,
    dt: float,
    dx: float,
    min_color_value: float,
    max_color_value: float,
    q_max: int,
    m_max: int,
    current_func: Callable[[int, xp.ndarray], None] | None = None,
    current_func_hash: str | None = None,
    J0: xp.ndarray | None = None,
    local_conductivity: xp.ndarray | None = None,
    perfect_conductor_mask: xp.ndarray | None = None,
    local_rel_permittivity: xp.ndarray | None = None,
    step_per_frame: int = 1,
    file_name: str | None = None,
    min_time_per_frame: int = 0,
    norm_type: str = "log",
    use_progress_bar: bool = True,
    precompute: bool = False,
    loop_animation: bool | None = None,
    show_from: int = 0,
    theme: str = "w",
    show_edges_of_materials: bool = True,
    show_material: bool = True,
) -> None:
    """Run the simulation and show the animation.
    This function will create a figure and an animation of the simulation.
    If a file_name is provided, the animation will not be shown,
    but saved to the file as an mp4.

    Args:
        E0 (xp.ndarray):
            [V/m] 2D array of the initial values of the electric field
            in the z direction on the main grid
        B_tilde_0 (xp.ndarray):
            [T] 2D array of the initial values of the magnetic field in both directions
        dt (float):
            [s] time step
        dx (float):
            [m] space step
        min_color_value (float):
            deprecated
        max_color_value (float):
            maximum color value for the log norm
        q_max (int):
            maximimum number of time steps
        m_max (int):
            number of space samples per dimension
        current_func (Callable[[int], xp.ndarray] | None, optional):
            function to get the current density in [A/m^2]. Defaults to None.
        J0 (xp.ndarray | None, optional):
            [A/m^2] 2D array of the initial values of the current density.
            Defaults to None.
        local_conductivity (xp.ndarray | None, optional):
            map of the local conductivity value. Defaults to None.
        perfect_conductor_mask (xp.ndarray | None, optional):
            mask of the perfect conductor region. Defaults to None.
        local_rel_permittivity (xp.ndarray | None, optional):
            map of the local relative permittivity value. Defaults to None.
        step_per_frame (int, optional):
            number of time steps per frame. Defaults to 1.
        file_name (str | None, optional):
            name of the file to save the animation, if given the animation won't show.
            Defaults to None.
        min_time_per_frame (int, optional):
            minimum time per frame in milliseconds. Defaults to 0.
        norm_type (str, optional):
            type of normalization to use, implemented options "log", "abslin", "lin".
            Defaults to "log".
        use_progress_bar (bool, optional):
            Whether to use a progress bar for the image_generation
            (only works for the first time showing the image). Defaults to False.
        precompute (bool, optional):
            Whether to precompute all the frames and show them at once to enable scrolling.
            can be memory intensive. for large simulation.
            Defaults to False.
        loop_animation (bool, optional):
            Whether to loop the animation. Defaults to True.
        show_from (int, optional):
            The time step from which to show the animation. Defaults to 0.
        theme (str, optional):
            The theme to use for the plot either "black" ("b") or "white" ("w"). Defaults to "w".
    """
    # check the norm type
    match norm_type:
        case "log":
            # logarithmic scale from 0 to 1
            scale = np.logspace(np.log10(min_color_value), 0, 255)
            show_abs = True
            levels = (0, max_color_value)
            color_map_name = "magma"
        case "lin" | "linear":
            scale = np.linspace(0, 1, 255)
            show_abs = False
            levels = (-max_color_value, max_color_value)
            color_map_name = "berlin"
        case "abslin":
            scale = np.linspace(0, 1, 255)
            show_abs = True
            levels = (0, max_color_value)
            color_map_name = "magma"
        case _:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    match theme:
        case "black" | "b":
            pass
        case "white" | "w":
            pg.setConfigOption("background", "w")
            pg.setConfigOption("foreground", "k")

    # clear temp directory
    os.makedirs("temp", exist_ok=True)
    for file in os.listdir("temp"):
        if file.endswith(".png"):
            os.remove(os.path.join("temp", file))

    if file_name is not None:
        if loop_animation:
            raise ValueError("loop_animation cannot be True if file_name is provided")
        loop_animation = False
    else:
        if loop_animation is None:
            loop_animation = True

    if J0 is None:
        J0 = xp.zeros((m_max, m_max), dtype=xp.float32)

    # transform the matplotlib colormap to a pyqtgraph colormap
    base_color_map: pg.ColorMap = pg.colormap.get(color_map_name, source="matplotlib")  # type: ignore

    base_color_map_lookuptable = base_color_map.getLookupTable(nPts=255)

    color_map = pg.ColorMap(
        scale, base_color_map_lookuptable, mapping=pg.ColorMap.MIRROR
    )

    def init():
        # initialise the arrays (only one instance saved, they will be updated in place)
        if show_from == 0:
            E[:, :] = xp.array(copy.deepcopy(E0))
            B_tilde_x[:, :] = xp.array(copy.deepcopy(B_tilde_0))
            B_tilde_y[:, :] = xp.array(copy.deepcopy(B_tilde_0))
        else:
            E[:, :] = xp.array(copy.deepcopy(E_ini))
            B_tilde_x[:, :] = xp.array(copy.deepcopy(B_tilde_x_ini))
            B_tilde_y[:, :] = xp.array(copy.deepcopy(B_tilde_y_ini))

    if show_from > 0:
        E_ini, B_tilde_x_ini, B_tilde_y_ini = simulate_up_to(
            dt,
            dx,
            show_from,
            m_max,
            current_func=current_func,
            local_conductivity=local_conductivity,
            local_rel_permittivity=local_rel_permittivity,
            perfect_conductor_mask=perfect_conductor_mask,
            J0=J0,
            E0=E0,
            B_tilde_x_0=B_tilde_0,
            B_tilde_y_0=B_tilde_0,
        )

    # created here to not interfere with the things printed by simulate_up_to
    match use_progress_bar:
        case True:
            temp = tqdm(range(show_from, q_max, step_per_frame), unit="f")
            temp.set_description("Generating image")
            frames = temp.__iter__()
        case False:
            temp = range(show_from, q_max, step_per_frame)
            frames = temp.__iter__()

    # allocate the arrays
    E: xp.ndarray = xp.ones((m_max, m_max), dtype=xp.float32) * 1e-30
    B_tilde_x = xp.ones((m_max, m_max), dtype=xp.float32) * 1e-30
    B_tilde_y = xp.ones((m_max, m_max), dtype=xp.float32) * 1e-30
    J = xp.ones((m_max, m_max), dtype=xp.float32) * 1e-30
    q = 0
    init()

    if using_cupy:
        # allocate the material arrays on the GPU
        if local_rel_permittivity is not None:
            local_rel_permittivity = xp.array(local_rel_permittivity)
        if local_conductivity is not None:
            local_conductivity = xp.array(local_conductivity)

    def update(image: pg.ImageItem):
        nonlocal q, frames, E, B_tilde_x, B_tilde_y, timer, plot
        try:
            q = frames.__next__()
        except StopIteration:
            if loop_animation:
                # drop the progress bar even if it was used for animation repeat
                frames = range(show_from, q_max, step_per_frame)
                frames = frames.__iter__()
                init()
                return
            else:
                # FIXME : find a way to start the creation of the video if specified
                timer.stop()
                return

        step_yee(
            E,
            B_tilde_x,
            B_tilde_y,
            J,
            q,
            dt,
            dx,
            local_rel_permittivity,
            current_func,
            perfect_conductor_mask,
            local_conductivity,
        )

        plot.setTitle(f"Electric field in the z direction at time {q * dt:.2e} s")

        if show_abs:
            image.setImage(
                xp.abs(E),
                autoLevels=False,
                autoRange=False,
            )
        else:
            image.setImage(
                E,
                autoLevels=False,
                autoRange=False,
            )

        if file_name is not None:
            # save the image to a file
            pyqtgraph.exporters.ImageExporter(plot).export(
                os.path.join("temp", f"frame_{q}.png")
            )

    # initialise plotting
    pyqtgraph.setConfigOptions(useCupy=using_cupy)
    widget = pg.GraphicsLayoutWidget()
    widget.setWindowTitle("FDTD 2D Yee algorithm")
    widget.resize(1000, 900)  # FIXME can't get the ImageItem to resize properly
    widget.show()

    plot = widget.addPlot(
        title="Electric field in the z direction at time 0 s",
    )
    im = pg.ImageItem(
        E,
        autoLevels=False,
        levels=levels,
        axisOrder="row-major",
    )
    im.setColorMap(color_map)
    im.setRect(0, 0, 400, 400)  # FIXME can't get the ImageItem to resize properly
    plot.addItem(im)
    plot.showAxes(True)  # frame it with a full set of axes
    plot.invertY(True)

    # add a colorbar
    color_bar_label = ""
    match norm_type:
        case "log":
            color_bar_label = "log(Ez)"
        case "lin":
            color_bar_label = "Ez [V/m]"
        case "abslin":
            color_bar_label = "abs(Ez) [V/m]"
        case _:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    color_bar = pg.ColorBarItem(
        values=levels,
        label=color_bar_label,
    )
    color_bar.setImageItem(im, insert_in=plot)

    if show_material:
        material_image = create_material_image(
            local_conductivity,
            local_rel_permittivity,
            perfect_conductor_mask,
            m_max,
            show_material=show_material,
            show_edges_of_materials=show_edges_of_materials,
        )
        mat_im = pg.ImageItem(
            material_image,
            axisOrder="row-major",
        )
        mat_im.setRect(0, 0, 400, 400)
        plot.addItem(mat_im)
        mat_im.setZValue(10)  # put it on top of the other image
        

    im.setColorMap(color_map)
    timer = QtCore.QTimer()  # type: ignore
    timer.timeout.connect(lambda: update(im))
    timer.start(min_time_per_frame)

    pg.exec()


def simulate_and_plot(
    ax: Axes,
    dt: float,
    dx: float,
    Q: int,
    m_max: int,
    current_func: Callable[[int, xp.ndarray], None] | None,
    norm_type: str = "log",
    local_conductivity: np.ndarray | None = None,
    local_rel_permittivity: np.ndarray | None = None,
    perfect_conductor_mask: np.ndarray | None = None,
    J0: np.ndarray | None = None,
    E0: np.ndarray | None = None,
    B_tilde_x_0: np.ndarray | None = None,
    B_tilde_y_0: np.ndarray | None = None,
    use_progress_bar: bool = True,
    color_bar=True,
    min_color_value: float | None = None,
    show_edges_of_materials: bool = True,
    show_material: bool = True,
) -> Tuple[AxesImage, np.ndarray]:
    """
    Simulate the Yee algorithm and plot the results on the given matplotlib.axes.Axes object.

    Args:
        dt (float): _description_
        dx (float): _description_
        Q (int): _description_
        m_max (int): _description_
        current_func (Callable[[int, xp.ndarray], None] | None): _description_
        norm_type (str, optional): _description_. Defaults to "log".
        local_conductivity (np.ndarray | None, optional): _description_. Defaults to None.
        local_rel_permittivity (np.ndarray | None, optional): _description_. Defaults to None.
        perfect_conductor_mask (np.ndarray | None, optional): _description_. Defaults to None.
        J0 (np.ndarray | None, optional): _description_. Defaults to None.
        E0 (np.ndarray | None, optional): _description_. Defaults to None.
        B_tilde_0 (np.ndarray | None, optional): _description_. Defaults to None.
        use_progress_bar (bool, optional): _description_. Defaults to True.

    Returns:
        AxesImage: _description_
    """
    E_z, _, _ = simulate_up_to(
        dt,
        dx,
        Q,
        m_max,
        current_func,
        local_conductivity,
        local_rel_permittivity,
        perfect_conductor_mask,
        J0,
        E0,
        B_tilde_x_0,
        B_tilde_y_0,
        use_progress_bar=use_progress_bar,
    )

    if using_cupy and not TYPE_CHECKING:
        E_z = xp.asnumpy(E_z)

    material_image = create_material_image(
        local_conductivity,
        local_rel_permittivity,
        perfect_conductor_mask,
        m_max,
        show_material=show_material,
        show_edges_of_materials=show_edges_of_materials,
    )

    im = plot_field(
        ax,
        dx,
        E_z,
        image_overlay=material_image,
        min_color_value=min_color_value,
        norm_type=norm_type,
        color_bar=color_bar,
    )

    return im, E_z


def simulate_up_to(
    dt: float,
    dx: float,
    Q: int,
    m_max: int,
    current_func: Callable[[int, xp.ndarray], None] | None,
    local_conductivity: np.ndarray | None = None,
    local_rel_permittivity: np.ndarray | None = None,
    perfect_conductor_mask: np.ndarray | None = None,
    J0: np.ndarray | None = None,
    E0: np.ndarray | None = None,
    B_tilde_x_0: np.ndarray | None = None,
    B_tilde_y_0: np.ndarray | None = None,
    use_progress_bar: bool = True,
    return_numpy: bool = False,
) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Simulate the Yee algorithm and return the electric field E after Q time steps.

    Args:
        dt (float): _description_
        dx (float): _description_
        Q (int): _description_
        m_max (int): _description_
        current_func_hash (str | None): _description_
        current_func (Callable[[int, xp.ndarray], None] | None): _description_
        local_conductivity (np.ndarray | None, optional): _description_. Defaults to None.
        local_rel_permittivity (np.ndarray | None, optional): _description_. Defaults to None.
        perfect_conductor_mask (np.ndarray | None, optional): _description_. Defaults to None.
        J0 (np.ndarray | None, optional): _description_. Defaults to None.
        E0 (np.ndarray | None, optional): _description_. Defaults to None.
        B_tilde_0 (np.ndarray | None, optional): _description_. Defaults to None.
        use_progress_bar (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple[xp.ndarray, xp.ndarray, xp.ndarray]: Electric field E after Q time steps
    """

    # set the python hash function seed
    file_name = None
    hasher = xxhash.xxh32()

    if current_func is not None:
        # generate 5 steps of the current function to get a hash
        J = xp.zeros((m_max, m_max), dtype=np.float32)
        for q in range(5):
            current_func(q, J)
            hasher.update(J.tobytes())

    # start_time = time.perf_counter()
    # add all the parameters to the hash
    hasher.update(str(dt).encode("utf-8"))
    hasher.update(str(dx).encode("utf-8"))
    hasher.update(str(m_max).encode("utf-8"))

    if local_conductivity is not None:
        hasher.update(local_conductivity.tobytes())
    if local_rel_permittivity is not None:
        hasher.update(local_rel_permittivity.tobytes())
    if perfect_conductor_mask is not None:
        hasher.update(perfect_conductor_mask.tobytes())
    if J0 is not None:
        hasher.update(J0.tobytes())
    if E0 is not None:
        hasher.update(E0.tobytes())
    if B_tilde_x_0 is not None:
        hasher.update(B_tilde_x_0.tobytes())
    if B_tilde_y_0 is not None:
        hasher.update(B_tilde_y_0.tobytes())
    parameters_hash = hasher.hexdigest()
    # end_time = time.perf_counter()
    # print(f"Hashing parameters took {end_time - start_time:.4f} seconds")
    hasher.reset()
    # print(f"Hash of the parameters: {parameters_hash}")
    file_name = f"temp/simulate_{parameters_hash}_{Q}.npz"

    file_already_cached = False

    if file_name is not None and os.path.exists(file_name):
        loaded = np.load(file_name)
        E = loaded["E"]
        B_tilde_x = loaded["B_tilde_x"]
        B_tilde_y = loaded["B_tilde_y"]
        print(f"Loaded arrays from {file_name}")
        file_already_cached = True
    else:
        # try to load a previous step of the same simulation
        # find file name that match the pattern simulate_<hash>_<Q>.npz
        # and has a Q smaller than the current Q
        # get the list of files in the temp directory
        files = os.listdir("temp")
        # filter the files to keep only the ones that match the pattern
        files = [
            f
            for f in files
            if f.startswith(f"simulate_{parameters_hash}_") and f.endswith(".npz")
        ]
        # sort the files by Q
        files = sorted(
            files,
            key=lambda f: int(f.split("_")[-1].split(".")[0]),
            reverse=True,
        )
        # find the first file that has a Q smaller than the current Q
        loaded_Q = 0
        for f in files:
            file_Q = int(f.split("_")[-1].split(".")[0])
            if file_Q < Q:
                loaded_Q = file_Q
                # load the file
                loaded = np.load(os.path.join("temp", f))
                E0 = loaded["E"]
                B_tilde_x_0 = loaded["B_tilde_x"]
                B_tilde_y_0 = loaded["B_tilde_y"]
                print(f"Loaded intermediary step's arrays from {f}")
                break

        # initialize the arrays and move them to the GPU if using cupy
        E: xp.ndarray = xp.ones((m_max, m_max), dtype=xp.float32) * 1e-30
        B_tilde_x = xp.ones((m_max, m_max), dtype=xp.float32) * 1e-30
        B_tilde_y = xp.ones((m_max, m_max), dtype=xp.float32) * 1e-30
        J = xp.ones((m_max, m_max), dtype=xp.float32) * 1e-30
        q = 0

        if E0 is not None:
            E = xp.array(E0.copy())
        if B_tilde_x_0 is not None:
            B_tilde_x = xp.array(B_tilde_x_0.copy())
        if B_tilde_y_0 is not None:
            B_tilde_y = xp.array(B_tilde_y_0.copy())

        if local_conductivity is not None:
            local_conductivity = xp.array(local_conductivity)
        if local_rel_permittivity is not None:
            local_rel_permittivity = xp.array(local_rel_permittivity)
        if perfect_conductor_mask is not None:
            perfect_conductor_mask = xp.array(perfect_conductor_mask)

        match use_progress_bar:
            case True:
                steps = tqdm(range(loaded_Q, Q), unit="step")
                steps.set_description("Simulating")
            case False:
                steps = range(loaded_Q, Q)

        # compute the electric field after Q time steps
        for q in steps:
            step_yee(
                E,
                B_tilde_x,
                B_tilde_y,
                J,
                q,
                dt,
                dx,
                local_rel_permittivity,
                current_func,
                perfect_conductor_mask,
                local_conductivity,
            )

    # save the electric field to a file with a hash of the parameters as the name
    if not file_already_cached and file_name is not None:
        if using_cupy and not TYPE_CHECKING:
            E = xp.asnumpy(E)
            B_tilde_x = xp.asnumpy(B_tilde_x)
            B_tilde_y = xp.asnumpy(B_tilde_y)
        np.savez_compressed(file_name, E=E, B_tilde_x=B_tilde_x, B_tilde_y=B_tilde_y)
        # compute the file size
        file_size = os.path.getsize(file_name)
        file_size_mb = file_size / (1024 * 1024)
        print(f"Saved arrays to {file_name} using {file_size_mb:.2f} MB")

    if using_cupy and not TYPE_CHECKING and not return_numpy:
        # convert the arrays back to cupy arrays
        E = xp.array(E)
        B_tilde_x = xp.array(B_tilde_x)
        B_tilde_y = xp.array(B_tilde_y)
    else:
        # convert the arrays to numpy arrays
        # might not work to be tested
        E = np.array(E)
        B_tilde_x = np.array(B_tilde_x)
        B_tilde_y = np.array(B_tilde_y)

    return E, B_tilde_x, B_tilde_y


def compute_electric_field_amplitude(
    dt: float,
    dx: float,
    Q: int,
    m_max: int,
    period: float,
    current_func: Callable[[int, xp.ndarray], None] | None,
    local_conductivity: np.ndarray | None = None,
    local_rel_permittivity: np.ndarray | None = None,
    perfect_conductor_mask: np.ndarray | None = None,
    J0: np.ndarray | None = None,
    E0: np.ndarray | None = None,
    B_tilde_x_0: np.ndarray | None = None,
    B_tilde_y_0: np.ndarray | None = None,
    use_progress_bar: bool = True,
) -> np.ndarray:
    """Compute the electric field amplitude from the electric field in the z direction.

    Returns:
        np.ndarray: Electric field amplitude
    """
    E_z, B_tilde_x, B_tilde_y = simulate_up_to(
        dt,
        dx,
        Q,
        m_max,
        current_func,
        local_conductivity,
        local_rel_permittivity,
        perfect_conductor_mask,
        J0,
        E0,
        B_tilde_x_0,
        B_tilde_y_0,
        use_progress_bar=use_progress_bar,
    )

    J_z = xp.zeros((m_max, m_max), dtype=np.float32)

    num_steps = math.ceil(period / (2 * dt))
    E_amplitude = xp.zeros((m_max, m_max), dtype=np.float32)
    for i in tqdm(
        range(num_steps), unit="step", desc="Computing electric field amplitude"
    ):
        step_yee(
            E_z,
            B_tilde_x,
            B_tilde_y,
            J_z,
            i + Q,
            dt,
            dx,
            local_rel_permittivity,
            current_func,
            perfect_conductor_mask,
            local_conductivity,
        )
        E_amplitude = xp.maximum(E_amplitude, xp.abs(E_z))

    if using_cupy and not TYPE_CHECKING:
        E_amplitude = xp.asnumpy(E_amplitude)

    return E_amplitude


def compute_electric_field_amplitude_and_plot(
    ax: Axes,
    dt: float,
    dx: float,
    Q: int,
    m_max: int,
    period: float,
    current_func: Callable[[int, xp.ndarray], None] | None,
    norm_type: str = "log",
    local_conductivity: np.ndarray | None = None,
    local_rel_permittivity: np.ndarray | None = None,
    perfect_conductor_mask: np.ndarray | None = None,
    J0: np.ndarray | None = None,
    E0: np.ndarray | None = None,
    B_tilde_x_0: np.ndarray | None = None,
    B_tilde_y_0: np.ndarray | None = None,
    use_progress_bar: bool = True,
    color_bar=True,
    min_color_value: float | None = 1e-1,
    show_edges_of_materials: bool = True,
    show_material: bool = True,
) -> Tuple[AxesImage, np.ndarray]:
    """
    Compute the electric field amplitude and plot the results on the given matplotlib.axes.Axes object.
    """

    E_amp = compute_electric_field_amplitude(
        dt,
        dx,
        Q,
        m_max,
        period,
        current_func,
        local_conductivity,
        local_rel_permittivity,
        perfect_conductor_mask,
        J0,
        E0,
        B_tilde_x_0,
        B_tilde_y_0,
        use_progress_bar=use_progress_bar,
    )

    material_image = create_material_image(
        local_conductivity,
        local_rel_permittivity,
        perfect_conductor_mask,
        m_max,
        show_material=show_material,
        show_edges_of_materials=show_edges_of_materials,
    )

    im = plot_field(
        ax,
        dx,
        E_amp,
        image_overlay=material_image,
        min_color_value=min_color_value,
        norm_type=norm_type,
        color_bar=color_bar,
    )

    return im, E_amp


def plot_field(
    ax: Axes,
    dx: float,
    field: np.ndarray,
    image_overlay: np.ndarray | None = None,
    min_color_value: float | None = 0.1,
    max_color_value: float | None = None,
    norm_type: str = "log",
    color_bar: bool = True,
    color_bar_label: str | None = None,
) -> AxesImage:
    # check the norm type
    match norm_type:
        case "log":
            norm = LogNorm(vmin=min_color_value, vmax=max_color_value)
            show_abs = True
            color_map_name = "magma"
        case "abslin":
            norm = Normalize(vmin=min_color_value, vmax=max_color_value)
            show_abs = True
            color_map_name = "magma"
        case "lin":
            if max_color_value is None:
                max_color_value = np.max(np.abs(field), axis=None)
            norm = Normalize(vmin=-max_color_value, vmax=max_color_value)  # type: ignore
            show_abs = False
            color_map_name = "berlin"
        case _:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    # plot E as an image

    if show_abs:
        field_plot = np.abs(field)
        if color_bar_label is None:
            color_bar_label = "|Ez| [V/m]"
    else:
        field_plot = field
        if color_bar_label is None:
            color_bar_label = "Ez [V/m]"

    im = ax.imshow(
        field_plot,
        cmap=color_map_name,
        norm=norm,
        interpolation="nearest",
    )
    if color_bar:
        plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, label=color_bar_label)

    y_max = field.shape[0]
    x_max = field.shape[1]
    x_number_of_ticks = 10
    y_number_of_ticks = 10
    if y_max > x_max:
        x_number_of_ticks = int(x_number_of_ticks * (x_max / y_max))
    elif y_max < x_max:
        y_number_of_ticks = int(y_number_of_ticks * (y_max / x_max))

    unit = "m"
    scale_factor = 1
    if (x_max - 1) * dx / x_number_of_ticks < 1 or (
        y_max - 1
    ) * dx / y_number_of_ticks < 1:
        scale_factor = 1000
        unit = "mm"

    ax.set_xticks(
        np.linspace(0, x_max, x_number_of_ticks),
        np.round(
            np.linspace(0, (x_max - 1) * dx * scale_factor, x_number_of_ticks)
        ).astype(int),
    )
    ax.set_yticks(
        np.linspace(0, y_max, y_number_of_ticks),
        np.round(
            np.linspace(0, (y_max - 1) * dx * scale_factor, y_number_of_ticks)
        ).astype(int),
    )
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")

    if image_overlay is not None:
        ax.imshow(
            image_overlay,
        )

    return im


def field_to_power(
    E_z_amplitude: np.ndarray,
    R_a: float,
    h_e: float,
) -> np.ndarray:
    """Computes the power received by an antenna with the given parameters in each point of the grid.

    Args:
        E_z_amplitude (np.ndarray): The amplitude of the electric field in steady state in [V/m]
        R_a (float): Antenna resistance in [Ohm]
        h_e (float): Antenna equivalent height in [m] along the z axis (assumed omnidirectional in the xy plane)

    Returns:
        np.ndarray: power array in [W]
    """
    return 1 / 8 * h_e**2 * E_z_amplitude**2 / R_a


def get_exponential_decay_alpha(
    sigma: float,
    epsilon_r: float,
    mu_r: float,
    omega: float,
) -> float:
    """Computes the attenuation coefficient of the wave in a lossy medium.

    Args:
        sigma (float): Conductivity in [S/m]
        epsilon_r (float): Relative permittivity
        mu_r (float): Relative permeability
        omega (float): Angular frequency in [rad/s]

    Returns:
        float: attenuation coefficient in [1/m]
    """
    alpha = (
        omega
        * np.sqrt(e0 * epsilon_r * u0 * mu_r / 2)
        * np.sqrt(np.sqrt(1 + (np.max(sigma) / (omega * e0 * epsilon_r)) ** 2) - 1)
    )
    if using_cupy and not TYPE_CHECKING:
        alpha = xp.asnumpy(alpha)

    return alpha


def compute_poynting_integral(
    E_z: xp.ndarray,
    B_tilde_x: xp.ndarray,
    B_tilde_y: xp.ndarray,
    dx: float,
    upper_left_corner: Tuple[float, float],
    lower_right_corner: Tuple[float, float],
) -> float:
    """
    Computes the Poynting integral over the given square.
    Args:
        E_z (xp.ndarray): in [V/m] 2D array of the values of the electric field in the z direction on the main grid
        B_tilde_x (xp.ndarray): in [T] 2D array of the values of the magnetic field in the x direction on a grid shifted by 1/2 in the y direction
        B_tilde_y (xp.ndarray): in [T] 2D array of the values of the magnetic field in the y direction on a grid shifted by 1/2 in the x direction
    """
    # get the indices of the square
    x_min = round(upper_left_corner[0] / dx)
    x_max = round(lower_right_corner[0] / dx)
    y_min = round(upper_left_corner[1] / dx)
    y_max = round(lower_right_corner[1] / dx)

    # axis :
    #         ---> x
    #       |
    #       v y
    # E_z[y, x]

    # right side with 1n = 1x
    side_pos_1x = (
        -1
        / (2 * u0 * C_VIDE)
        * xp.sum(
            E_z[y_min:y_max, x_max]
            * (B_tilde_y[y_min:y_max, x_max - 1] + B_tilde_y[y_min:y_max, x_max])
        )
        * dx
    )

    side_neg_1x = (
        1
        / (2 * u0 * C_VIDE)
        * xp.sum(
            E_z[y_min:y_max, x_min]
            * (B_tilde_y[y_min:y_max, x_min - 1] + B_tilde_y[y_min:y_max, x_min])
        )
        * dx
    )

    side_pos_1y = (
        1
        / (2 * u0 * C_VIDE)
        * xp.sum(
            E_z[y_max, x_min:x_max]
            * (B_tilde_x[y_max - 1, x_min:x_max] + B_tilde_x[y_max, x_min:x_max])
        )
    )

    side_neg_1y = (
        -1
        / (2 * u0 * C_VIDE)
        * xp.sum(
            E_z[y_min, x_min:x_max]
            * (B_tilde_x[y_min - 1, x_min:x_max] + B_tilde_x[y_min, x_min:x_max])
        )
    )

    # compute the Poynting integral
    poynting_integral = side_pos_1x + side_neg_1x + side_pos_1y + side_neg_1y

    return poynting_integral


def compute_mean_poynting_integral(
    dt: float,
    dx: float,
    Q: int,
    m_max: int,
    period: float,
    current_func: Callable[[int, xp.ndarray], None] | None,
    upper_left_corner: Tuple[float, float],
    lower_right_corner: Tuple[float, float],
    local_conductivity: np.ndarray | None = None,
    local_rel_permittivity: np.ndarray | None = None,
    perfect_conductor_mask: np.ndarray | None = None,
    J0: np.ndarray | None = None,
    E0: np.ndarray | None = None,
    B_tilde_x_0: np.ndarray | None = None,
    B_tilde_y_0: np.ndarray | None = None,
    use_progress_bar: bool = True,
) -> float:
    """
    Compute the mean Poynting integral over a period.
    """
    E_z, B_tilde_x, B_tilde_y = simulate_up_to(
        dt,
        dx,
        Q,
        m_max,
        current_func,
        local_conductivity,
        local_rel_permittivity,
        perfect_conductor_mask,
        J0,
        E0,
        B_tilde_x_0,
        B_tilde_y_0,
        use_progress_bar=use_progress_bar,
    )

    J_z = xp.zeros((m_max, m_max), dtype=np.float32)

    num_steps = math.ceil(period / dt)
    mean_poynting_integral = 0
    for i in tqdm(
        range(num_steps), unit="step", desc="Computing mean Poynting integral"
    ):
        step_yee(
            E_z,
            B_tilde_x,
            B_tilde_y,
            J_z,
            i + Q,
            dt,
            dx,
            local_rel_permittivity,
            current_func,
            perfect_conductor_mask,
            local_conductivity,
        )
        mean_poynting_integral += (
            compute_poynting_integral(
                E_z,
                B_tilde_x,
                B_tilde_y,
                dx,
                upper_left_corner,
                lower_right_corner,
            )
            / num_steps
        )

    return float(mean_poynting_integral)
