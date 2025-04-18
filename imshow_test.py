import numpy as np
import pyqtgraph as pg

A = np.random.rand(100, 100, 100)

pg.image(A)

if __name__ == "__main__":
    pg.exec()