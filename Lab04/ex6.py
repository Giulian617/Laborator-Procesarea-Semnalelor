import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def ex6(file_path):
    sample_rate, signal = wavfile.read(file_path)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    group_length = int(0.1 * sample_rate)
    jump = group_length // 2

    groups = [signal[start:(start + group_length)]
              for start in range(0, len(signal)-group_length, jump)]

    fft_groups = [np.fft.fft(group) for group in groups]
    spectrogram = 20 * np.log10(np.abs(fft_groups) + 1e-9)

    plt.title("Spectrograma")
    plt.imshow(spectrogram.T, aspect="auto", origin="lower", cmap="viridis",
               extent=[0, len(signal)/sample_rate, 0, sample_rate // 2])
    plt.colorbar(label="Magnitudine")
    plt.savefig("Plot_ex6.pdf")


ex6("ex6.wav")
