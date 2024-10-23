import numpy as np
import matplotlib.pyplot as plt
import math


def obtain_color(dist):
    return np.array([0, 1-dist, 0, 1])


def generate_colors(samples):
    return np.array([obtain_color(np.sqrt(sample.real ** 2 + sample.imag ** 2))
                     for sample in samples])


def ex2_a(freq, no_samples):
    samples = np.linspace(0, 1, no_samples)

    fig, axs = plt.subplots(ncols=2, figsize=(14, 7))
    fig.suptitle(f"Semnal sinusoidal de frecventa {freq}Hz")

    axs[0].plot(samples, np.sin(2 * np.pi * freq * samples))

    values = np.sin(2 * np.pi * freq * samples) * \
        math.e**(-2 * np.pi * 1j * samples)
    axs[1].scatter(values.real, values.imag, color=generate_colors(values))

    axs[1].set_xlim(-1.25, 1.25)
    axs[1].set_ylim(-1.25, 1.25)
    axs[1].set_aspect("equal")

    plt.savefig("Plot_ex2_a.pdf")
    plt.clf()


def ex2_b(freq, no_samples, omega):
    samples = np.linspace(0, 1, no_samples)

    fig, axs = plt.subplots(len(omega), figsize=(7, len(omega)*7))
    fig.suptitle(
        f"Frecventa de infasurare pentru un semnal sinusoidal de frecventa {freq}Hz")

    for i in range(len(omega)):
        values = np.sin(2 * np.pi * freq * samples) * \
            math.e**(-2 * np.pi * 1j * omega[i] * samples)
        axs[i].scatter(values.real, values.imag, color=generate_colors(values))

        axs[i].set_xlim(-1.25, 1.25)
        axs[i].set_ylim(-1.25, 1.25)
        axs[i].set_aspect("equal")

    plt.savefig("Plot_ex2_b.pdf")
    plt.clf()


ex2_a(7, 1000)
ex2_b(7, 3000, [1, 4, 7, 10])
