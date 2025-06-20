# FDTD - Finite-Difference Time-Domain
# 1D simulation using Yee's algorithm
# to simulate electromagnetic waves.


from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# parameters
# settings parameters
TOTAL_X = 10  # in meters
FREQ_REF = 1e8  # Hz
Q = 20000  # number of time samples
COND_START = 1  # in meters
COND_END = 7  # in meters
CONDUCTIVITY = 0.003  # in S/m
REL_PERMITTIVITY = 1  # in F/m
SOURCE_POS = 9  # in meters


# Constants
e0 = 8.8541878188e-12  # F/m
u0 = 1.25663706127e-6  # H/m
c_vide = 1 / np.sqrt(e0 * u0)  # m/s


# derived parameters
DELTA_X = c_vide / (FREQ_REF * 100)  # in meters
DELTA_T = 1 / (2 * FREQ_REF * 100)  # in seconds
# REMARK : when DELTA_T is too small(comparend to DELTA_x), the limit conditions seam to stop working correctly (a 10x difference causes problems)
# the current limit condition assumes that C * DELTA_T/ DELTA_X = 2 (? I found a source that says 1 but I'm not sure : https://opencourses.emu.edu.tr/pluginfile.php/2641/mod_resource/content/1/ABC.pdf)
M = round(TOTAL_X / DELTA_X)  # number of space samples

# set the local relative permittivity array
epsilon_r = np.ones((M))
# epsilon_r[int(3 * M / 4) - 25 : int(3 * M / 4) + 25] = 4
# slowest speed in the medium
c_slowest = 1 / np.sqrt(np.max(epsilon_r) * e0 * u0)


# set the local conductivity array
sigma = np.zeros((M))

start_cond = round(COND_START / DELTA_X)
end_cond = round(COND_END / DELTA_X)
sigma[start_cond:end_cond] = CONDUCTIVITY
epsilon_r[start_cond:end_cond] = REL_PERMITTIVITY

TOTAL_T = (Q - 1) * DELTA_T  # in seconds
print("TOTAL_X : ", TOTAL_X, "TOTAL_T : ", TOTAL_T)
print("DELTA_X : ", DELTA_X, "DELTA_T : ", DELTA_T)

# check the stability condition
print("OLD feedback current max multiplier : ", (DELTA_T / e0) * np.max(sigma))


# %%
# initialise the starting values
E0 = np.zeros((M))
B_tilde_0 = np.zeros((M))
# intiialise the current density
J_source = np.zeros((Q, M))

# set a sinusoidal current at the middle of the grid
fraction_on = 1
J_source[0 : int(Q * fraction_on), round(SOURCE_POS / DELTA_X)] = (
    0.01
    / DELTA_X  # insures that the total current is constant
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
def forward_E(E: np.ndarray, B_tilde: np.ndarray, J_source: np.ndarray, q: int):
    """
    modifies E in place at step q
    q : int : has to be between 1 and Q-1
    """
    E[q, 1:M] = (1 / (1 + sigma[1:M] * DELTA_T / (epsilon_r[1:M] * e0*2))) * (
        E[q - 1, 1:M] * (1 - sigma[1:M] * DELTA_T / (epsilon_r[1:M] * e0*2))
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


main()


fig, (ax1) = plt.subplots()
ax1: Axes = ax1


x = np.linspace(0, TOTAL_X, M) - COND_END
ax1.set_xlim(np.min(x), np.max(x))
ax1.set_xlabel("x (m)")
ax1.set_ylabel("E (V/m)")
ax1.set_title(f"Amplitude du champ électrique en régime sinusoïdal permanent")
ax1.tick_params(axis="y", labelcolor="b")
plt.ylim(
    -0.1 * (np.max(E, axis=None) - np.min(E, axis=None)),
    np.max(E, axis=None) + 0.1 * (np.max(E, axis=None) - np.min(E, axis=None)),
)

ax1.axvspan(
    start_cond * DELTA_X - COND_END,
    end_cond * DELTA_X - COND_END,
    color="gray",
    alpha=0.5,
)


steps_per_half_period = int(1 / (2 * FREQ_REF * DELTA_T))

print(
    "steps_per_half_period : ",
    steps_per_half_period,
    "approx : ",
    1 / (2 * FREQ_REF * DELTA_T),
)

amplitude = np.max(np.abs(E[-steps_per_half_period:, :]), axis=0)
ax1.plot(x, amplitude, label="Amplitude Ez", color="red")

# visualisation de la décroissance exponentielle théorique
omega = 2 * np.pi * FREQ_REF
alpha = (
    omega
    * np.sqrt(e0 * REL_PERMITTIVITY * u0 / 2)
    * np.sqrt(np.sqrt(1 + (np.max(sigma) / (omega * e0 * REL_PERMITTIVITY)) ** 2) - 1)
)
print(f"alpha : {alpha:.2e} m^-1")
print(f"skin depth : {1 / alpha:.2e} m")

A0 = amplitude[end_cond - 1]


x_in_dielectric = x[start_cond:end_cond]
theoretical_decay = A0 * np.exp(-alpha * (-(x_in_dielectric - x_in_dielectric[-1])))
ax1.plot(
    x_in_dielectric,
    theoretical_decay,
    label="Theoretical decay",
    color="orange",
)

ax1.legend(loc="upper left")
plt.savefig("images/1D_conductivity_validation.png", dpi=300, bbox_inches="tight")

plt.show()
