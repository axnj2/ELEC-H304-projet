# FDTD - Finite-Difference Time-Domain
# 2D simulation using Yee's algorithm
# to simulate electromagnetic waves.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import copy

from yee_FDTD_2D import step_yee, e0, c_vide
from current_sources import sinusoïdal_point_source

# parameters
# settings parameters
M = 1001  # number of space samples per dimension
FREQ_REF = 1e8  # Hz
Q = 10000  # number of time samples
TOTAL_CURRENT = 0.01  # A
INITIAL_ZERO = 0  # initial value for E and B_tilde
MIN_COLOR = 1e-6  # minimum color value for the image


# derived parameters
DELTA_X = c_vide / (FREQ_REF * 80)  # in meters
DELTA_T = 1 / (2 * FREQ_REF * 80)  # in seconds
all_time_max = 10 * TOTAL_CURRENT / (DELTA_X * DELTA_X) * DELTA_T / e0
def current_func(q:int) -> np.ndarray : 
    return sinusoïdal_point_source(
        q,
        M,
        M // 2,
        M // 2,
        TOTAL_CURRENT,
        FREQ_REF,
        DELTA_T,
        DELTA_X,
    )

TOTAL_X = (M - 1) * DELTA_X  # in meters
TOTAL_T = (Q - 1) * DELTA_T  # in seconds
print("TOTAL_X : ", TOTAL_X, "TOTAL_T : ", TOTAL_T)
print("DELTA_X : ", DELTA_X, "DELTA_T : ", DELTA_T)

# %%
# initialise the starting values
E0 = np.ones((M, M)) * INITIAL_ZERO
B_tilde_0 = np.ones((M, M)) * INITIAL_ZERO


# %%


# latex equation around the point [m,n], m,n in [1, M-1[   :
# \begin{align}
# E_z^{q+1}[m,n]=&E_z^{q}[m,n] \\
# &+\frac{\Delta t}{\varepsilon_0 \mu_0}  \cdot( \\
# &- \frac{B_x^{q+1 / 2}[m, n+1 / 2]-B_x^{q+1 / 2}[m,n-1 / 2]}{\Delta y}  \\
# &+ \frac{B_y^{q+1 / 2}[m+1 / 2, n]-B_y^{q+1 / 2}[m-1 / 2,n]}{\Delta x} \\
# &) - \mu_{0} J_z^{q+1 / 2}[m,n]
# \end{align}
# but we are using B_tilde = c*B so we have to replace B by B_tilde/c in the equation


# latex equations around the point [m,n], m,n in [0, M[   :
# B_x^{q+1 / 2}[m, n+1 / 2]= B_x^{q-1 / 2}[m,n+1 / 2]+\frac{\Delta t}{\Delta y}\left(E_z^q[m,n+1]-E_z^q[m,n]\right)
# B_y^{q+1 / 2}[m+1 / 2, n]= B_y^{q-1 / 2}[m+1 / 2,n]+\frac{\Delta t}{\Delta x}\left(E_z^q[m+1,n]-E_z^q[m,n]\right)
# but we are using B_tilde = c*B so we have to replace B by B_tilde/c in the equation


# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html

fig, ax1 = plt.subplots()
fig.set_size_inches(8, 8)

initial_image = MIN_COLOR * np.ones((M, M))

im = ax1.imshow(
    initial_image,
    interpolation="nearest",
    origin="lower",
    cmap="jet",
    norm=LogNorm(vmin=MIN_COLOR, vmax=all_time_max),
)

fig.colorbar(im, ax=ax1, orientation="vertical", pad=0.01)


E = np.zeros((M, M))
B_tilde_x = np.zeros((M, M))
B_tilde_y = np.zeros((M, M))


def init():
    # initialise the arrays (only one instance saved, they will be updated in place)
    E[:, :] = copy.deepcopy(E0)
    B_tilde_x[:, :] = copy.deepcopy(B_tilde_0)
    B_tilde_y[:, :] = copy.deepcopy(B_tilde_0)
    return (im,)


init()


def update(q: int):
    step_yee(
        E,
        B_tilde_x,
        B_tilde_y,
        q,
        DELTA_T,
        DELTA_X,
        current_func,
    )
    im.set_data(np.abs(E))

    return (im,)


ani = animation.FuncAnimation(
    fig, update, frames=range(1, Q), interval=0, blit=True, init_func=init
)

plt.show()
