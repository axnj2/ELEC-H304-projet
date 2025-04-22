import numpy as np

from typing import (
    Callable,
    Iterable,
)  # https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints

import pyqtgraph as pg

import copy

from tqdm import tqdm

from pyqtgraph.Qt import QtCore


# Constants
e0: float = 8.8541878188e-12  # F/m
u0: float = 1.25663706127e-6  # H/m
C_VIDE: float = 1 / np.sqrt(e0 * u0)  # m/s


def forward_E(
    E: np.ndarray,
    B_tilde_x: np.ndarray,
    B_tilde_y: np.ndarray,
    q: int,
    M: int,
    dt: float,
    dx: float,
    J: np.ndarray,
    epsilon_r: np.ndarray | None,
    local_conductivity: np.ndarray | None,
):
    """Performs a time step forward for the electric field E.
    Assumes a square grid

    Args:
        E (np.ndarray): in [V/m] 2D array of the values of the electric field in the z direction on the main grid
        B_tilde_x (np.ndarray): in [T] 2D array of the values of the magnetic field in the x direction on a grid shifted by 1/2 in the y direction
        B_tilde_y (np.ndarray): in [T] 2D array of the values of the magnetic field in the y direction on a grid shifted by 1/2 in the x direction
        dt (float, optional): time step in [s]. Defaults to DELTA_T.
        dx (float, optional): space step in [m]. Defaults to DELTA_X.
        q (int): time step number
        current_source_func (Callable[[int], np.ndarray], optional): function to get the current density in [A/m^2]. Defaults to get_source_J.
        epsilon_r (np.ndarray | None, optional): map of the local relative permittivity value. Defaults to None.
    """

    # set the epsilon_r to 1 if not provided
    if epsilon_r is None:
        epsilon_r = np.ones((M, M), dtype=np.float32)

    # set the local conductivity to 0 if not provided
    if local_conductivity is None:
        local_conductivity = np.zeros((M, M), dtype=np.float32)

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
    E[-1, :] = np.ones((M), dtype=np.float32) * 1e-300
    E[:, -1] = np.ones((M), dtype=np.float32) * 1e-300
    E[0, :] = np.ones((M), dtype=np.float32) * 1e-300
    E[:, 0] = np.ones((M), dtype=np.float32) * 1e-300


def forward_B_tilde(
    E: np.ndarray,
    B_tilde_x: np.ndarray,
    B_tilde_y: np.ndarray,
    M: int,
    dt: float,
    dx: float,
):
    """Performs a time step forward for the magnetic field B_tilde.
    Assumes a square grid

    Args:
        E (np.ndarray): in [V/m] 2D array of the values of the electric field in the z direction on the main grid
        B_tilde_x (np.ndarray): in [T] 2D array of the values of the magnetic field in the x direction on a grid shifted by 1/2 in the y direction
        B_tilde_y (np.ndarray): in [T] 2D array of the values of the magnetic field in the y direction on a grid shifted by 1/2 in the x direction
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
    E: np.ndarray,
    B_tilde_x: np.ndarray,
    B_tilde_y: np.ndarray,
    q: int,
    dt: float,
    dx: float,
    epsilon_r: np.ndarray | None,
    current_source_func: Callable[[int], np.ndarray] | None,
    perferct_conductor_mask: np.ndarray | None,
    local_conductivity: np.ndarray | None,
):
    """Performs a time step for the Yee algorithm.
    This function updates the electric field E and the magnetic field B_tilde_x
    and B_tilde_y
    using the Yee algorithm.

    Will set the electric field back to zero in the perfect conductor region.

    It assumes a square grid.

    Args:
        E (np.ndarray): in [V/m] 2D array of the values of the electric field in the z direction on the main grid
        B_tilde_x (np.ndarray): in [T] 2D array of the values of the magnetic field in the x direction on a grid shifted by 1/2 in the y direction
        B_tilde_y (np.ndarray): in [T] 2D array of the values of the magnetic field in the y direction on a grid shifted by 1/2 in the x direction
        q (int): time step number
        dt (float, optional): time step in [s]. Defaults to DELTA_T.
        dx (float, optional): space step in [m]. Defaults to DELTA_X.
        current_source_func (Callable[[int], np.ndarray], optional): function to get the current density in [A/m^2]. Defaults to get_source_J.
        perferct_conductor_mask (np.ndarray | None, optional): mask of the perfect conductor region. Defaults to None.
    """
    # infer the grid size
    M = E.shape[0]

    # get the current density
    if current_source_func is not None:
        J = current_source_func(q)
    else:
        J = np.zeros((M, M), dtype=np.float32)

    # validate that the dimensions are coeherent
    assert E.shape[0] == E.shape[1], "Error: E must be a square matrix"
    assert E.shape == B_tilde_x.shape, "Error: E and B_tilde_x must have the same shape"
    assert E.shape == B_tilde_y.shape, "Error: E and B_tilde_y must have the same shape"
    if perferct_conductor_mask is not None:
        assert E.shape == perferct_conductor_mask.shape, (
            "Error: E and perfect conductor mask must have the same shape"
        )
    if current_source_func is not None:
        assert E.shape == current_source_func(0).shape, (
            "Error: E and J must have the same shape"
        )

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
    E0: np.ndarray,
    B_tilde_0: np.ndarray,
    dt: float,
    dx: float,
    min_color_value: float,
    max_color_value: float,
    q_max: int,
    m_max: int,
    current_func: Callable[[int], np.ndarray] | None = None,
    local_conductivity: np.ndarray | None = None,
    perfect_conductor_mask: np.ndarray | None = None,
    local_rel_permittivity: np.ndarray | None = None,
    step_per_frame: int = 1,
    file_name: str | None = None,
    min_time_per_frame: int = 0,
    norm_type: str = "log",
    use_progress_bar: bool = True,
    precompute: bool = False,
    loop_animation: bool = True,
) -> None:
    """Run the simulation and show the animation.
    This function will create a figure and an animation of the simulation.
    If a file_name is provided, the animation will not be shown,
    but saved to the file as an mp4.

    Args:
        E0 (np.ndarray):
            [V/m] 2D array of the initial values of the electric field
            in the z direction on the main grid
        B_tilde_0 (np.ndarray):
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
        current_func (Callable[[int], np.ndarray] | None, optional):
            function to get the current density in [A/m^2]. Defaults to None.
        local_conductivity (np.ndarray | None, optional):
            map of the local conductivity value. Defaults to None.
        perfect_conductor_mask (np.ndarray | None, optional):
            mask of the perfect conductor region. Defaults to None.
        local_rel_permittivity (np.ndarray | None, optional):
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
    """
    # check the norm type
    match norm_type:
        # TODO : add log and abs lin
        case "log":
            # logarithmic scale from 0 to 1
            scale = np.logspace(-2, 0, 512)
            show_abs = True
            levels = (0, max_color_value)
        case "lin":
            scale = np.linspace(0, 1, 512)
            show_abs = False
            levels = (-max_color_value, max_color_value)
        case "abslin":
            scale = np.linspace(0, 1, 512)
            show_abs = True
            levels = (0, max_color_value)
        case _:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    
    match use_progress_bar:
        case True:
            frames = tqdm(range(1, q_max // step_per_frame), unit="f")
            frames.set_description("Generating image")
            frames = frames.__iter__()
        case False:
            frames = range(1, q_max // step_per_frame)
            frames = frames.__iter__()

    
    base_color_map: pg.ColorMap = pg.colormap.get("magma")  # type: ignore

    base_color_map_lookuptable = base_color_map.getLookupTable(nPts=512)

    color_map = pg.ColorMap(
        scale, base_color_map_lookuptable, mapping=pg.ColorMap.MIRROR
    )

    def init():
        # initialise the arrays (only one instance saved, they will be updated in place)
        E[:, :] = copy.deepcopy(E0)
        B_tilde_x[:, :] = copy.deepcopy(B_tilde_0)
        B_tilde_y[:, :] = copy.deepcopy(B_tilde_0)




    # allocate the arrays
    E = np.zeros((m_max, m_max), dtype=np.float32)
    B_tilde_x = np.zeros((m_max, m_max), dtype=np.float32)
    B_tilde_y = np.zeros((m_max, m_max), dtype=np.float32)
    q = 0

    def update(image: pg.ImageItem):
        nonlocal q, frames, E, B_tilde_x, B_tilde_y, timer
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
                timer.stop()
                return
            
        step_yee(
            E,
            B_tilde_x,
            B_tilde_y,
            q,
            dt,
            dx,
            local_rel_permittivity,
            current_func,
            perfect_conductor_mask,
            local_conductivity,
        )
        if show_abs:
            image.setImage(
                np.abs(E),
                autoLevels=False,
            )
        else:
            image.setImage(
                E,
                autoLevels=False,
            )

    initial_image = min_color_value * np.ones((m_max, m_max))

    if precompute:
        all_E = np.zeros((q_max, m_max, m_max))
        for q in frames:
            step_yee(
                E,
                B_tilde_x,
                B_tilde_y,
                q,
                dt,
                dx,
                local_rel_permittivity,
                current_func,
                perfect_conductor_mask,
                local_conductivity,
            )
            all_E[q] = E

        if show_abs:
            im = pg.image(
                np.abs(all_E),
            )
        else:
            im = pg.image(
                all_E,
            )
        

        im.setColorMap(color_map)

    else:
        im = pg.image(
            initial_image,
            levels=levels,
        )
        im.setColorMap(color_map)

        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: update(im))
        timer.start(min_time_per_frame)

    pg.exec()
