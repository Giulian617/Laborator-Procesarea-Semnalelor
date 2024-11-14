import numpy as np
import matplotlib.pyplot as plt


def rectangular_window(Nw):
    return np.ones(Nw)


def hanning_window(Nw):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(Nw) / Nw))


def ex3(freq, no_samples, Nw):
    samples = np.linspace(0, 1, no_samples)[:Nw]

    fig, axs = plt.subplots(2, figsize=(10, 8))
    axs[0].plot(samples, np.sin(2 * np.pi * freq * samples)
                * rectangular_window(Nw))
    axs[1].plot(samples, np.sin(2 * np.pi * freq * samples)
                * hanning_window(Nw))

    plt.savefig("Plot_ex3.pdf")


ex3(100, 2000, 200)
