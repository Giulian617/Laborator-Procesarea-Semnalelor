import numpy as np
import matplotlib.pyplot as plt


def d(n, fs):
    data = np.genfromtxt('Train.csv', delimiter=',')
    x = [int(data[i][2]) for i in range(1, len(data))]

    fourier = np.fft.fft(x)/np.sqrt(n)
    freq = fs*np.linspace(0, n/2-1, n//2)

    plt.title(f"Transformata Fourier")
    plt.plot(freq, np.abs(fourier[:n//2]))
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("Plot_d.pdf")


d(18288, 1/3600)
