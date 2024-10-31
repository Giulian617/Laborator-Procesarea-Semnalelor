import numpy as np
import matplotlib.pyplot as plt
import math
import time
import json
import os


def my_linspace(start, finish, no_samples):
    distance = (finish-start)/no_samples
    return np.array([start + i * distance for i in range(no_samples)])


def ex1(testing_sizes):
    def dft(signal, n):
        fourier = np.matrix(np.zeros((n, n), dtype=complex))
        for i in range(n):
            for j in range(n):
                fourier[i, j] = math.e**(1j * (-2) * np.pi * i * j / n)
        return np.matmul(fourier, signal)

    if not os.path.exists("DFTvsFFT.json"):
        file = open("DFTvsFFT.json", "w")

        dft_json = {}
        fft_json = {}

        for i in range(len(testing_sizes)):
            samples = my_linspace(
                0, 1, testing_sizes[i])
            signal = np.sin(2 * np.pi * 1 * samples)

            start = time.perf_counter()
            dft(signal, testing_sizes[i])
            end = time.perf_counter()
            dft_json[int(testing_sizes[i])] = end - start

            start = time.perf_counter()
            np.fft.fft(signal)
            end = time.perf_counter()
            fft_json[int(testing_sizes[i])] = end - start

        json.dump({"DFT": dft_json, "FFT": fft_json}, file, indent=4)

    with open("DFTvsFFT.json") as file:
        json_content = json.load(file)

        dft_times = json_content["DFT"]
        fft_times = json_content["FFT"]

        dft_results = [dft_times[key] for key in dft_times]
        fft_results = [fft_times[key] for key in fft_times]

        fig, axs = plt.subplots(2)
        fig.suptitle("DFT vs FFT")
        axs[0].plot(testing_sizes, dft_results)
        axs[1].plot(testing_sizes, fft_results)

        axs[0].set_yscale("log")
        axs[1].set_yscale("log")

        plt.savefig("Plot_ex1.pdf")


ex1(np.array([128, 256, 512, 1024, 2048, 4096, 8192]))
