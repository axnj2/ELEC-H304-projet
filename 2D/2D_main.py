# FDTD - Finite-Difference Time-Domain
# 2D simulation using Yee's algorithm
# to simulate electromagnetic waves.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
DELTA_X = c_vide / (FREQ_REF * 10)  # in meters
DELTA_T = 1 / (2 * FREQ_REF * 10)  # in seconds

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

    J[round(M / 2), round(M / 2)] = np.sin(2 * np.pi * FREQ_REF * q * DELTA_T)
    return J

# initialise the arrays
E = np.zeros((Q, M, M))
B_tilde = np.zeros((Q, M, M))
E[0, :, :] = E0
B_tilde[0, :, :] = B_tilde_0


