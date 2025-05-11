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

import copy

from tqdm import tqdm

from matplotlib import pyplot as plt

def forward_B_tilde(
    M: int,
    DELTA_T: float,
    DELTA_X: float,
    E_x: xp.ndarray,
    E_y: xp.ndarray,
    E_z: xp.ndarray,
    B_tilde_x: xp.ndarray,
    B_tilde_y: xp.ndarray,
    B_tilde_z: xp.ndarray,
) -> None:
    """
    Update the magnetic field using the Yee algorithm.

    Args:
        M (int): number of space samples per dimension
        DELTA_T (float): time step
        DELTA_X (float): space step
        E_x (xp.ndarray): electric field in x direction
        E_y (xp.ndarray): electric field in y direction
        E_z (xp.ndarray): electric field in z direction
        B_tilde_x (xp.ndarray): magnetic field in x direction
        B_tilde_y (xp.ndarray): magnetic field in y direction
        B_tilde_z (xp.ndarray): magnetic field in z direction
    """
    

def step_yee(
    M: int,
    q: int,
    DELTA_T: float,
    DELTA_X: float,
    E_x: xp.ndarray,
    E_y: xp.ndarray,
    E_z: xp.ndarray,
    B_tilde_x: xp.ndarray,
    B_tilde_y: xp.ndarray,
    B_tilde_z: xp.ndarray,
    J_x: xp.ndarray,
    J_y: xp.ndarray,
    J_z: xp.ndarray,
    source: Callable,
) -> None:
    """
    Perform one step of the Yee algorithm.

    Args:
        M (int): number of space samples per dimension
        DELTA_T (float): time step
        DELTA_X (float): space step
        E_x (xp.ndarray): electric field in x direction
        E_y (xp.ndarray): electric field in y direction
        E_z (xp.ndarray): electric field in z direction
        B_tilde_x (xp.ndarray): magnetic field in x direction
        B_tilde_y (xp.ndarray): magnetic field in y direction
        B_tilde_z (xp.ndarray): magnetic field in z direction
        J_x (xp.ndarray): current density in x direction
        J_y (xp.ndarray): current density in y direction
        J_z (xp.ndarray): current density in z direction
    """
    source(q, J_x, J_y, J_z)

    # update the magnetic field
    forward_B_tilde(
        M, DELTA_T, DELTA_X, E_x, E_y, E_z, B_tilde_x, B_tilde_y, B_tilde_z
    )
    # update the electric field
    forward_E(
        M, DELTA_T, DELTA_X, E_x, E_y, E_z, B_tilde_x, B_tilde_y, B_tilde_z, J_x, J_y, J_z
    )

def simulate_and_animate(M: int, Q: int, DELTA_T: float, DELTA_X: float, source: Callable) -> None:


    # alocate the memory for the fields
    E_x = xp.zeros((M, M, M), dtype=xp.float32)
    E_y = xp.zeros((M, M, M), dtype=xp.float32)
    E_z = xp.zeros((M, M, M), dtype=xp.float32)
    B_tilde_x = xp.zeros((M, M, M), dtype=xp.float32)
    B_tilde_y = xp.zeros((M, M, M), dtype=xp.float32)
    B_tilde_z = xp.zeros((M, M, M), dtype=xp.float32)
    J_x = xp.zeros((M, M, M), dtype=xp.float32)
    J_y = xp.zeros((M, M, M), dtype=xp.float32)
    J_z = xp.zeros((M, M, M), dtype=xp.float32)




