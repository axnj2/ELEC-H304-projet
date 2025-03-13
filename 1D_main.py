# FDTD - Finite-Difference Time-Domain
# 1D simulation using Yee's algorithm
# to simulate electromagnetic waves.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# parameters
# settings parameters
M = 400  # number of space samples
FREQ_REF = 1e8  # Hz
Q = 10000  # number of time samples


# Constants
e0 = 8.8541878188e-12  # F/m
u0 = 1.25663706127e-6  # H/m
c_vide = 1 / np.sqrt(e0 * u0)  # m/s

# set the local relative permittivity array
epsilon_r = np.ones((M))
# epsilon_r[int(3 * M / 4) - 25 : int(3 * M / 4) + 25] = 4

# set the local conductivity array
sigma = np.zeros((M))
sigma[300:320] = 1

# derived parameters
DELTA_X = c_vide / (FREQ_REF * 40)  # in meters
DELTA_T = 1 / (2 * FREQ_REF * 400)  # in seconds

TOTAL_X = (M - 1) * DELTA_X  # in meters
TOTAL_T = (Q - 1) * DELTA_T  # in seconds
print("TOTAL_X : ", TOTAL_X, "TOTAL_T : ", TOTAL_T)
print("DELTA_X : ", DELTA_X, "DELTA_T : ", DELTA_T)

# check the stability condition
print("feedback current max multiplier : ", (DELTA_T /  e0) * np.max(sigma))
assert ((DELTA_T /  e0) * np.max(sigma)<1), "stability condition not met"


# %%
# initialise the starting values
E0 = np.zeros((M))
B_tilde_0 = np.zeros((M))
# intiialise the current density
J_source = np.zeros((Q, M))

# initialise with a gaussian pulse
# E0[int(M / 2) - 100 : int(M / 2) + 100] = np.exp(
#     -(np.linspace(-100, 100, 200) ** 2) /100
# ) / np.sqrt(2 * np.pi)

# set a sinusoidal current at the middle of the grid
fraction_on = 1
J_source[0 : int(Q * fraction_on), round(M / 2)] = (
    1
    / 10
    * np.sin(
        2
        * np.pi
        * FREQ_REF
        * np.linspace(0, TOTAL_T * fraction_on, int(Q * fraction_on))
    )
)


# initialise the arrays
E = np.zeros((Q, M))
E[0, :] = E0[:]
B_tilde = np.zeros((Q, M))
B_tilde[0, :] = B_tilde_0[:]


# %%
# We consider B_tilde to be half a time step in front of E in the time domain,
# this means that we computing step n for E we use step n-1 for B
# we also consider B_tilde's grid half a step ahead of E's grid in the space domain
# this means that x=0 for E <-> x=0.5 for B_tilde
def forward_E(E: np.array, B_tilde: np.array, J_source: np.array, q: int):
    """
    modifies E in place at step q
    q : int : has to be between 1 and Q-1
    """
    E[q, 1 : M - 1] = (
        E[q - 1, 1 : M - 1]
        + (1 / (epsilon_r[1 : M - 1] * e0 * u0))
        * (DELTA_T / ((1 / np.sqrt(epsilon_r[1 : M - 1] * e0 * u0)) * DELTA_X))
        * (B_tilde[q - 1, 1 : M - 1] - B_tilde[q - 1, 0 : M - 2])
        - (DELTA_T / (epsilon_r[1 : M - 1] * e0))
        * (J_source[q - 1, 1 : M - 1] + sigma[1 : M - 1] * E[q - 1, 1 : M - 1])
    )


    # limit conditions :
    E[q, 0] = E[q - 2, 1]
    E[q, M - 1] = E[q - 2, M - 2]


def forward_B_tilde(E: np.array, B_tilde: np.array, q: int):
    """
    modifies B_tilde in place at step q
    q : int : the time step : has to be between 1 and Q-1
    """
    # limit conditions :

    B_tilde[q, 0 : M - 1] = B_tilde[q - 1, 0 : M - 1] + (
        ((1 / np.sqrt(epsilon_r[0 : M - 1] * e0 * u0)) * DELTA_T) / DELTA_X
    ) * (E[q, 1:M] - E[q, 0 : M - 1])

    # limit conditions :
    B_tilde[q, M - 1] = B_tilde[q - 2, M - 2]
    B_tilde[q, 0] = B_tilde[q - 2, 1]


def main():
    for q in range(1, Q):
        forward_E(E, B_tilde, J_source, q)
        forward_B_tilde(E, B_tilde, q)


# %%
main()


# %%
# animate the results : https://stackoverflow.com/questions/67672601/how-to-use-matplotlibs-animate-function
fig, ax1 = plt.subplots()
ax1.set_xlim(0, TOTAL_X)

x = np.linspace(0, TOTAL_X, M)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("E (V/m)")
ax1.set_title("1D FDTD simulation")
ax1.tick_params(axis="y", labelcolor="b")
(lineE,) = plt.plot(x, E[0], label="0 s", color="b")
(lineJ,) = plt.plot(x, J_source[0], label="0 s", color="g")
plt.legend()
plt.ylim(
    np.min(E, axis=None),
    np.max(E, axis=None) + 0.1 * (np.max(E, axis=None) - np.min(E, axis=None)),
)

# show the relative permittivity on the plot
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
ax2 = ax1.twinx()
ax2.plot(x, sigma, "r")
ax2.set_ylabel("conductivité", color="r")
ax2.tick_params(axis="y", labelcolor="r")

frame_devider = 5


def updatefig(i):
    lineE.set_ydata(E[i * frame_devider])
    lineE.set_label(f"{i * frame_devider * DELTA_T:.2e} s")
    lineJ.set_ydata(J_source[i * frame_devider])
    plt.legend()
    return (lineE, lineJ)


ani = animation.FuncAnimation(
    fig, updatefig, frames=int(Q / frame_devider), repeat=True, interval=1
)

#ani.save("1D_sine_source_local_conductivity.mp4", fps=60)

plt.show()
