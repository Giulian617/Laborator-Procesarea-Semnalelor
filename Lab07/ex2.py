import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage


def calculate_snr(original, current):
    noise = original - current
    P_signal = np.sum(original**2)
    P_noise = np.sum(noise**2)
    if P_noise == 0:
        return np.inf
    return P_signal / P_noise


def remove_noise(X, cutoff_ratio):
    Y = np.fft.fft2(X)
    freq_cutoff = np.max(np.abs(Y)) * cutoff_ratio
    Y[np.abs(Y) > freq_cutoff] = 0
    return np.real(np.fft.ifft2(Y))


def ex2(X, target_snr):
    iteration = 0
    curr_snr = np.inf
    X_copy = X.copy()

    print(f"At iteration {iteration} SNR = {curr_snr}")
    while target_snr < curr_snr and iteration <= 100:
        iteration += 1
        X = remove_noise(X, 0.8)
        curr_snr = calculate_snr(X_copy, X)
        print(f"At iteration {iteration} SNR = {curr_snr}")

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(X_copy, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(X, cmap="gray")
    axs[1].set_title("Final")

    plt.savefig("Plot2.pdf")


ex2(misc.face(gray=True), 0.0075)
