from typing import TYPE_CHECKING

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

import os


# Constants
e0: float = 8.8541878188e-12  # F/m
u0: float = 1.25663706127e-6  # H/m
C_VIDE: float = 1 / math.sqrt(e0 * u0)  # m/s


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

    print("min and max epsilon_r", xp.min(epsilon_r), xp.max(epsilon_r))
    # update the electric field
    E[1:M, 1:M] = (
        E[1:M, 1:M]
        + dt
        / (C_VIDE * e0 * epsilon_r[1:M, 1:M] * u0 * dx)
        * (
            -(B_tilde_x[1:M, 1:M] - B_tilde_x[0 : M - 1, 1:M])
            + (B_tilde_y[1:M, 1:M] - B_tilde_y[1:M, 0 : M - 1])
        )
        - dt
        / (e0 * epsilon_r[1:M, 1:M])
        * (J[1:M, 1:M] + local_conductivity[1:M, 1:M] * E[1:M, 1:M])
    )

    # set the boundary conditions
    E[-1, :] = xp.ones((M), dtype=xp.float32) * 1e-300
    E[:, -1] = xp.ones((M), dtype=xp.float32) * 1e-300
    E[0, :] = xp.ones((M), dtype=xp.float32) * 1e-300
    E[:, 0] = xp.ones((M), dtype=xp.float32) * 1e-300


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
        E[perferct_conductor_mask] = 1e-300


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
        # TODO : add log and abs lin
        case "log":
            # logarithmic scale from 0 to 1
            scale = np.logspace(-2, 0, 512)
            show_abs = True
            levels = (0, max_color_value)
            color_map_name = "magma"
        case "lin" | "linear":
            scale = np.linspace(0, 1, 512)
            show_abs = False
            levels = (-max_color_value, max_color_value)
            color_map_name = "berlin"
        case "abslin":
            scale = np.linspace(0, 1, 512)
            show_abs = True
            levels = (0, max_color_value)
            color_map_name = "magma"
        case _:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    match use_progress_bar:
        case True:
            temp = tqdm(range(1, q_max // step_per_frame), unit="f")
            temp.set_description("Generating image")
            frames = temp.__iter__()
        case False:
            temp = range(1, q_max // step_per_frame)
            frames = temp.__iter__()

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

    base_color_map_lookuptable = base_color_map.getLookupTable(nPts=512)

    color_map = pg.ColorMap(
        scale, base_color_map_lookuptable, mapping=pg.ColorMap.MIRROR
    )

    def init():
        # initialise the arrays (only one instance saved, they will be updated in place)
        E[:, :] = xp.array(copy.deepcopy(E0))
        B_tilde_x[:, :] = xp.array(copy.deepcopy(B_tilde_0))
        B_tilde_y[:, :] = xp.array(copy.deepcopy(B_tilde_0))
        J[:, :] = xp.array(copy.deepcopy(J0))

    # allocate the arrays
    E: xp.ndarray = xp.zeros((m_max, m_max), dtype=xp.float32)
    B_tilde_x = xp.zeros((m_max, m_max), dtype=xp.float32)
    B_tilde_y = xp.zeros((m_max, m_max), dtype=xp.float32)
    J = xp.zeros((m_max, m_max), dtype=xp.float32)
    q = 0

    def update(image: pg.ImageItem):
        nonlocal q, frames, E, B_tilde_x, B_tilde_y, timer, plot
        try:
            q = frames.__next__()
        except StopIteration:
            if loop_animation:
                # drop the progress bar even if it was used for animation repeat
                frames = range(1, q_max // step_per_frame)
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

        if q >= show_from:
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

    im.setColorMap(color_map)
    timer = QtCore.QTimer()  # type: ignore
    timer.timeout.connect(lambda: update(im))
    timer.start(min_time_per_frame)

    pg.exec()
