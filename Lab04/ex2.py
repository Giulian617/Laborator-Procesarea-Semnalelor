import numpy as np
import matplotlib.pyplot as plt
import math


def my_linspace(start, finish, no_samples):
    distance = (finish-start)/no_samples
    return np.array([start + i * distance for i in range(no_samples)])


def ex2(freq, fs):
    fig, axs = plt.subplots(3)
    fig.suptitle("Testare teorema Nyquist")

    samples1 = my_linspace(0, 1, 12)
    samples2 = my_linspace(0, 1, 1000)

    axs[0].plot(samples2, np.sin(2 * np.pi * 13 * samples2), color="orange")
    axs[0].stem(samples1, np.sin(2 * np.pi * 13 * samples1))
    axs[0].plot(samples1, np.sin(2 * np.pi * 13 * samples1))

    axs[1].plot(samples2, np.sin(2 * np.pi * 25 * samples2), color="orange")
    axs[1].stem(samples1, np.sin(2 * np.pi * 25 * samples1))
    axs[1].plot(samples1, np.sin(2 * np.pi * 25 * samples1))

    axs[2].plot(samples2, np.sin(2 * np.pi * 37 * samples2), color="orange")
    axs[2].stem(samples1, np.sin(2 * np.pi * 37 * samples1))
    axs[2].plot(samples1, np.sin(2 * np.pi * 37 * samples1))

    plt.savefig("Plot_ex2.pdf")


ex2([1, 9, 16], 8)
