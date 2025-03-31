# FDTD - Finite-Difference Time-Domain
# 2D simulation using Yee's algorithm
# to simulate electromagnetic waves.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

from pprint import pprint


# parameters
# settings parameters
M = 100  # number of space samples per dimension
FREQ_REF = 1e8  # Hz
Q = 500  # number of time samples


# Constants
e0 = 8.8541878188e-12  # F/m
u0 = 1.25663706127e-6  # H/m
c_vide = 1 / np.sqrt(e0 * u0)  # m/s

# derived parameters
DELTA_X = c_vide / (FREQ_REF * 20)  # in meters
DELTA_T = 1 / (2 * FREQ_REF * 80)  # in seconds

TOTAL_X = (M - 1) * DELTA_X  # in meters
TOTAL_T = (Q - 1) * DELTA_T  # in seconds
print("TOTAL_X : ", TOTAL_X, "TOTAL_T : ", TOTAL_T)
print("DELTA_X : ", DELTA_X, "DELTA_T : ", DELTA_T)

# %%
# initialise the starting values
E0 = np.zeros((M, M))
B_tilde_0 = np.zeros((M, M))


# create current density source function
def get_source_J(q):
    J = np.zeros((M, M))
    # set a sinusoidal current at the middle of the grid

    if q < Q / 10:
        J[round(M / 2), round(M / 2)] = 0.1 * np.sin(2 * np.pi * FREQ_REF * q * DELTA_T)
    return J


# initialise the arrays (only one instance saved, they will be updated in place)
E = np.zeros((M, M))
B_tilde_x = np.zeros((M, M))
B_tilde_y = np.zeros((M, M))
E = E0
B_tilde_x = copy.deepcopy(B_tilde_0)
B_tilde_y = copy.deepcopy(B_tilde_0)

# %%

"""
latex equation around the point [m,n], m,n in [1, M-1[   : 
\begin{align}
E_z^{q+1}[m,n]=&E_z^{q}[m,n] \\
&+\frac{\Delta t}{\varepsilon_0 \mu_0}  \cdot( \\
&- \frac{B_x^{q+1 / 2}[m, n+1 / 2]-B_x^{q+1 / 2}[m,n-1 / 2]}{\Delta y}  \\
&+ \frac{B_y^{q+1 / 2}[m+1 / 2, n]-B_y^{q+1 / 2}[m-1 / 2,n]}{\Delta x} \\
&) - \mu_{0} J_z^{q+1 / 2}[m,n]
\end{align}

but we are using B_tilde = c*B so we have to replace B by B_tilde/c in the equation
"""


def forward_E(E: np.ndarray, B_tilde_x: np.ndarray, B_tilde_y: np.ndarray, q: int):
    # get the current density
    J = get_source_J(q)

    # update the electric field
    E[1:M, 1:M] = (
        E[1:M, 1:M]
        + DELTA_T
        / (c_vide * e0 * u0 * DELTA_X)
        * (
            -(B_tilde_x[1:M, 1:M] - B_tilde_x[0 : M - 1, 1:M])
            + (B_tilde_y[1:M, 1:M] - B_tilde_y[1:M, 0 : M - 1])
        )
        - u0 * J[1:M, 1:M]
    )


"""
latex equations around the point [m,n], m,n in [0, M[   :
B_x^{q+1 / 2}[m, n+1 / 2]= B_x^{q-1 / 2}[m,n+1 / 2]+\frac{\Delta t}{\Delta y}\left(E_z^q[m,n+1]-E_z^q[m,n]\right)

B_y^{q+1 / 2}[m+1 / 2, n]= B_y^{q-1 / 2}[m+1 / 2,n]+\frac{\Delta t}{\Delta x}\left(E_z^q[m+1,n]-E_z^q[m,n]\right)

but we are using B_tilde = c*B so we have to replace B by B_tilde/c in the equation

"""


def forward_B_tilde(E: np.ndarray, B_tilde_x: np.ndarray, B_tilde_y: np.ndarray):
    # update the magnetic field
    B_tilde_x[0 : M - 1, 0 : M - 1] = B_tilde_x[
        0 : M - 1, 0 : M - 1
    ] + c_vide * DELTA_T / DELTA_X * (E[1:M, 0 : M - 1] - E[0 : M - 1, 0 : M - 1])

    B_tilde_y[0 : M - 1, 0 : M - 1] = B_tilde_y[
        0 : M - 1, 0 : M - 1
    ] + c_vide * DELTA_T / DELTA_X * (E[0 : M - 1, 1:M] - E[0 : M - 1, 0 : M - 1])


# %%

# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html

fig, ax1 = plt.subplots()

im = ax1.imshow(E, interpolation="nearest", origin="lower")


def update(q: int):
    forward_E(E, B_tilde_x, B_tilde_y, q)
    forward_B_tilde(E, B_tilde_x, B_tilde_y)
    im.set_data(np.abs(E))
    print(np.min(E, axis=None), np.max(E, axis=None))
    return (im,)


ani = animation.FuncAnimation(fig, update, frames=range(1, Q), interval=200)

plt.show()

# t = 0
# # %%

# forward_E(E, B_tilde_x, B_tilde_y, t)
# forward_B_tilde(E, B_tilde_x, B_tilde_y)
# print("t = ", t, "J = ", np.max(get_source_J(t), axis=None))
# pprint(E)
# pprint(B_tilde_x)
# pprint(B_tilde_y)
# t+= 1




# %%
