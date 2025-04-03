import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# generate a 2D sinusoidal wave starting from the center
def generate_wave(M, FREQ_REF, DELTA_X, DELTA_T, q):
    c = 3e8  # speed of light in m/s
    x = np.linspace(-M / 2, M / 2, M)
    y = np.linspace(-M / 2, M / 2, M)
    X, Y = np.meshgrid(x, y)
    wave = 10*np.sin(2 * np.pi * (c / FREQ_REF) * np.sqrt(X**2 + Y**2) / DELTA_X + 2* np.pi * FREQ_REF * q * DELTA_T)
    return wave

# create a figure and axis
fig, ax = plt.subplots()
# plot the image
M=100
wave = generate_wave(M, 1e8, 1e2, 1e-10, 0)
im = ax.imshow(np.zeros([M, M]),  interpolation='nearest')
# animate the image
def update(frame):
    wave = generate_wave(100, 1e8, 1e2, 1e-9, frame)
    im.set_array(wave)
    return im,

ani = animation.FuncAnimation(fig, update, frames=100, interval=60)

plt.show()