# FDTD - Finite-Difference Time-Domain
# 1D simulation using Yee's algorithm
# to simulate electromagnetic waves.


from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# parameters
# settings parameters
TOTAL_X = 4  # in meters
GAUSSIAN_TIME_WIDTH = 1e-9  # in seconds
Q = 10000  # number of time samples
COND_START = 2  # in meters
COND_END = 8  # in meters
CONDUCTIVITY = 0  # in S/mµ
DIEL_START = 25 - 8  # in meters
DIEL_END = 25 - 2  # in meters
DEIL__REL_PERMITTIVITY = 1
SOURCE_POS = TOTAL_X / 2  # in meters

alpha = 1 / (2 * GAUSSIAN_TIME_WIDTH**2)
FREQ_REF = np.sqrt(12 * alpha) * 2 * np.pi  # Hz (voir rapport pour la formule)
print(f"FREQ_REF : {FREQ_REF:.2e} Hz")
# Constants
e0 = 8.8541878188e-12  # F/m
u0 = 1.25663706127e-6  # H/m
c_vide = 1 / np.sqrt(e0 * u0)  # m/s


# derived parameters
DELTA_X = c_vide / (FREQ_REF * 20)  # in meters
DELTA_T = 1 / (2 * FREQ_REF * 20)  # in seconds
# REMARK : when DELTA_T is too small(comparend to DELTA_x), the limit conditions seam to stop working correctly (a 10x difference causes problems)
# the current limit condition assumes that C * DELTA_T/ DELTA_X = 2 (? I found a source that says 1 but I'm not sure : https://opencourses.emu.edu.tr/pluginfile.php/2641/mod_resource/content/1/ABC.pdf)
M = round(TOTAL_X / DELTA_X)  # number of space samples

# set the local relative permittivity array
epsilon_r = np.ones((M))
start_diel = round(DIEL_START / DELTA_X)
end_diel = round(DIEL_END / DELTA_X)
epsilon_r[start_diel:end_diel] = DEIL__REL_PERMITTIVITY
print(f"start_diel : {start_diel}, end_diel : {end_diel}, M : {M}")
# slowest speed in the medium
c_slowest = 1 / np.sqrt(np.max(epsilon_r) * e0 * u0)


# set the local conductivity array
sigma = np.zeros((M))

start_cond = round(COND_START / DELTA_X)
end_cond = round(COND_END / DELTA_X)
sigma[start_cond:end_cond] = CONDUCTIVITY

TOTAL_T = (Q - 1) * DELTA_T  # in seconds
print("TOTAL_X : ", TOTAL_X, "TOTAL_T : ", TOTAL_T)
print("DELTA_X : ", DELTA_X, "DELTA_T : ", DELTA_T)

# check the stability condition
print("feedback current max multiplier : ", (DELTA_T / e0) * np.max(sigma))
assert (DELTA_T / e0) * np.max(sigma) < 1, "stability condition not met"


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

J_source[0 : int(Q), round(SOURCE_POS / DELTA_X)] = (
    -0.01
    / DELTA_X  # insures that the total current is constant
    * np.exp(-1 * (np.linspace(0, TOTAL_T, int(Q))- 3*GAUSSIAN_TIME_WIDTH) ** 2 * alpha)
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
def forward_E(E: np.ndarray, B_tilde: np.ndarray, J_source: np.ndarray, q: int):
    """
    modifies E in place at step q
    q : int : has to be between 1 and Q-1
    """
    E[q, 1:M] = (1 / (1 + sigma[1:M] * DELTA_T / (epsilon_r[1:M] * e0))) * (
        E[q - 1, 1:M]
        + (1 / (epsilon_r[1:M] * e0 * u0))
        * (DELTA_T / (c_vide * DELTA_X))
        * (B_tilde[q - 1, 1:M] - B_tilde[q - 1, 0 : M - 1])
        - (DELTA_T / (epsilon_r[1:M] * e0)) * (J_source[q - 1, 1:M])
    )

    # limit conditions :
    E[q, 0] = E[q - 2, 1]
    E[q, M - 1] = E[q - 2, M - 2]


def forward_B_tilde(E: np.ndarray, B_tilde: np.ndarray, q: int):
    """
    modifies B_tilde in place at step q
    q : int : the time step : has to be between 1 and Q-1
    """
    B_tilde[q, 0 : M - 1] = B_tilde[q - 1, 0 : M - 1] + (
        (c_vide * DELTA_T) / DELTA_X
    ) * (E[q, 1:M] - E[q, 0 : M - 1])


def main():
    for q in range(1, Q):
        forward_E(E, B_tilde, J_source, q)
        forward_B_tilde(E, B_tilde, q)


# %%
main()


# %%
# animate the results : https://stackoverflow.com/questions/67672601/how-to-use-matplotlibs-animate-function
fig, (ax1) = plt.subplots()
ax1: Axes = ax1


x = np.linspace(0, TOTAL_X, M) - TOTAL_X / 2
ax1.set_xlim(np.min(x), np.max(x))
ax1.set_xlabel("x (m)")
ax1.set_ylabel("E (V/m)")
ax1.set_title(
    f"Impulsion gaussienne d'écart type {GAUSSIAN_TIME_WIDTH:.0e} s "
)
ax1.tick_params(axis="y", labelcolor="b")
(lineE,) = plt.plot(x, E[0], label="0 s", color="b")
(lineJ,) = plt.plot(x, J_source[0], label="0 s", color="g")
plt.ylim(
    np.min(E, axis=None),
    np.max(E, axis=None) + 0.1 * (np.max(E, axis=None) - np.min(E, axis=None)),
)

# ax1.axvspan(
#     start_cond * DELTA_X - TOTAL_X / 2,
#     end_cond * DELTA_X - TOTAL_X / 2,
#     color="gray",
#     alpha=0.5,
# )
# ax1.axvspan(
#     DIEL_START - TOTAL_X / 2,
#     DIEL_END - TOTAL_X / 2,
#     color="blue",
#     alpha=0.5,
#     label="dielectric",
# )

frame_devider = 1


def updatefig(i: int):
    lineE.set_ydata(E[i * frame_devider])
    lineE.set_label(f"Ez at {i * frame_devider * DELTA_T:.2e} s")
    lineJ.set_ydata(J_source[i * frame_devider])
    lineJ.set_label(f"J at {i * frame_devider * DELTA_T:.2e} s")
    plt.legend()
    return (lineE, lineJ)


# ani = animation.FuncAnimation(
#     fig, updatefig, frames=range(3000, Q), repeat=True, interval=0
# )


# steps_per_half_period = int(1 / (2 * FREQ_REF * DELTA_T))

# print(
#     "steps_per_half_period : ",
#     steps_per_half_period,
#     "approx : ",
#     1 / (2 * FREQ_REF * DELTA_T),
# )

# ax1.plot(
#     x,
#     np.max(np.abs(E[-steps_per_half_period:, :]), axis=0),
#     label="Amplitude Ez",
#     color="red",
# )
updatefig(4000)

# ani.save("1D_sine_source_local_conductivity.mp4", fps=60)
plt.legend(loc="lower left")
plt.savefig(
    "images/1D_gaussian_pulse.png", dpi=300, bbox_inches="tight"
)

plt.show()
