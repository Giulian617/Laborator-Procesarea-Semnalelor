import numpy as np
import matplotlib.pyplot as plt


def ex1(n):
    x = np.random.rand(n)
    fig, axs = plt.subplots(4)

    for ax in axs:
        ax.plot(x)
        x = np.convolve(x, x)
        x = x / np.max(x)

    plt.savefig("Plot_ex1.pdf")


ex1(100)
# La final, semnalul seamana ca o curba Gaussiana
