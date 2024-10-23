import numpy as np
import matplotlib.pyplot as plt
import math


def my_linspace(start, finish, no_samples):
    distance = (finish-start)/no_samples
    return np.array([start + i * distance for i in range(no_samples)])


def ex3(signal, samples):
    n = signal.shape[0]
    fourier = np.zeros(n//2, dtype=complex)
    for i in range(n//2):
        for j in range(n):
            fourier[i] += signal[j] * math.e ** (-2 * np.pi * 1j * j * i / n)

    fig, axs = plt.subplots(2, figsize=(7, 14))
    fig.suptitle(f"Transformata Fourier pentru un semnal dat")

    axs[0].plot(samples, signal)
    axs[1].stem(np.arange(n//2), np.abs(fourier))
    plt.savefig("Plot_ex3.pdf")
    plt.clf()


samples = my_linspace(0, 1, 512)
ex3(np.sin(2 * np.pi * 3 * samples) + 10 * np.sin(2 * np.pi *
    6 * samples) + 3 * np.sin(2 * np.pi * 17 * samples), samples)
