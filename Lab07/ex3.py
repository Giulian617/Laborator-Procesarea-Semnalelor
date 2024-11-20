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


def add_noise(X, pixel_noise):
    return X + np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)


def ex3(X, pixel_noise):
    X_noisy = add_noise(X, pixel_noise)
    X_filtered = ndimage.median_filter(X_noisy, size=7)

    print(f"Initial SNR = {calculate_snr(X, X_noisy)}")
    print(f"Final SNR = {calculate_snr(X_noisy, X_filtered)}")

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(X_noisy, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(X_filtered, cmap="gray")
    axs[1].set_title("Final")

    plt.savefig("Plot3.pdf")


ex3(misc.face(gray=True), 200)
