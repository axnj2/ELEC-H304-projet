import numpy as np
from typing import (
    Callable,
)  # https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import copy


# Constants
e0: float = 8.8541878188e-12  # F/m
u0: float = 1.25663706127e-6  # H/m
c_vide: float = 1 / np.sqrt(e0 * u0)  # m/s


def forward_E(
    E: np.ndarray,
    B_tilde_x: np.ndarray,
    B_tilde_y: np.ndarray,
    q: int,
    M: int,
    dt: float,
    dx: float,
    current_source_func: Callable[[int], np.ndarray] | None = None,
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
    """
    # get the current density
    if current_source_func is not None:
        J = current_source_func(q)
    else:
        J = np.zeros((M, M))

    # update the electric field
    E[1:M, 1:M] = (
        E[1:M, 1:M]
        + dt
        / (c_vide * e0 * u0 * dx)
        * (
            -(B_tilde_x[1:M, 1:M] - B_tilde_x[0 : M - 1, 1:M])
            + (B_tilde_y[1:M, 1:M] - B_tilde_y[1:M, 0 : M - 1])
        )
        - dt / e0 * J[1:M, 1:M]
    )

    # set the boundary conditions
    E[-1, :] = np.ones((M)) * 1e-30
    E[:, -1] = np.ones((M)) * 1e-30
    E[0, :] = np.ones((M)) * 1e-30
    E[:, 0] = np.ones((M)) * 1e-30


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
    ] - c_vide * dt / dx * (E[1:M, 0 : M - 1] - E[0 : M - 1, 0 : M - 1])

    B_tilde_y[0 : M - 1, 0 : M - 1] = B_tilde_y[
        0 : M - 1, 0 : M - 1
    ] + c_vide * dt / dx * (E[0 : M - 1, 1:M] - E[0 : M - 1, 0 : M - 1])


def step_yee(
    E: np.ndarray,
    B_tilde_x: np.ndarray,
    B_tilde_y: np.ndarray,
    q: int,
    dt: float,
    dx: float,
    current_source_func: Callable[[int], np.ndarray] | None = None,
    perferct_conductor_mask: np.ndarray | None = None,
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
    """

    # infer the grid size
    M = E.shape[0]
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

    forward_E(E, B_tilde_x, B_tilde_y, q, M, dt, dx, current_source_func)

    forward_B_tilde(E, B_tilde_x, B_tilde_y, M, dt, dx)


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
    perfect_conductor_mask: np.ndarray | None = None,
    file_name: str | None = None,
) -> None:
    """Run the simulation and 

    Args:
        E0 (np.ndarray): _description_
        B_tilde_0 (np.ndarray): _description_
        dt (float): _description_
        dx (float): _description_
        min_color_value (float): _description_
        max_color_value (float): _description_
        q_max (int): _description_
        m_max (int): _description_
        current_func (Callable[[int], np.ndarray] | None, optional): _description_. Defaults to None.
        perfect_conductor_mask (np.ndarray | None, optional): _description_. Defaults to None.
    """
    def init():
        # initialise the arrays (only one instance saved, they will be updated in place)
        E[:, :] = copy.deepcopy(E0)
        B_tilde_x[:, :] = copy.deepcopy(B_tilde_0)
        B_tilde_y[:, :] = copy.deepcopy(B_tilde_0)
        return (im,)

    def update(q: int):
        step_yee(
            E,
            B_tilde_x,
            B_tilde_y,
            q,
            dt,
            dx,
            current_func,
            perfect_conductor_mask,
        )
        im.set_data(np.abs(E))

        return (im,)

    # allocate the arrays
    E = np.zeros((m_max, m_max))
    B_tilde_x = np.zeros((m_max, m_max))
    B_tilde_y = np.zeros((m_max, m_max))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 8)

    initial_image = min_color_value * np.ones((m_max, m_max))

    im = ax1.imshow(
        initial_image,
        interpolation="nearest",
        origin="lower",
        cmap="jet",
        norm=LogNorm(vmin=min_color_value, vmax=max_color_value),
    )

    fig.colorbar(im, ax=ax1, orientation="vertical", pad=0.01)

    init()

    animation.FuncAnimation(
        fig, update, frames=range(1, q_max), interval=0, blit=True, init_func=init
    )

    plt.show()
