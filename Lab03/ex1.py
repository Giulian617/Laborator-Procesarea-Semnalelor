import numpy as np
import matplotlib.pyplot as plt
import math


def ex1(n):
    fourier = np.matrix(np.zeros((n, n), dtype=complex))
    for i in range(n):
        for j in range(n):
            fourier[i, j] = math.e**(1j * (-2) * np.pi * i * j / n)

    product = fourier * fourier.getH()

    if np.allclose(product, n * np.identity(n)):
        fig, axs = plt.subplots(n)
        fig.suptitle("DFT matrix")
        coordinates = np.arange(n)
        fourier_array = np.array(fourier)

        for i in range(n):
            axs[i].plot(coordinates, fourier_array[i, :].real, color="orange")
            axs[i].plot(coordinates, fourier_array[i, :].imag, color="blue")

        plt.savefig("Plot_ex1.pdf")
        plt.clf()


ex1(8)
