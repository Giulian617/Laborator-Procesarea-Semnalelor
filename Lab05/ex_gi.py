import numpy as np
import matplotlib.pyplot as plt
import datetime


def gi(n, fs, idx, filter_value):
    data = np.genfromtxt('Train.csv', delimiter=',', dtype=str)[1:]
    dates = [datetime.datetime.strptime(
        date, '%d-%m-%Y %H:%M') for date in data[:, 1]]

    if dates[idx].weekday() == 0:
        idx -= int(dates[idx].hour)
    while dates[idx].weekday() != 0:
        idx += 1

    x = [int(data[i][2]) for i in range(idx, idx + 24*30)]

    plt.title(f"Semnalul incepand cu ziua {idx}, timp de o luna")
    plt.plot(np.arange(24*30), x)
    plt.savefig("Plot_g.pdf")
    plt.clf()

    fourier = np.fft.fft(x)/np.sqrt(n)
    fourier = [a if np.abs(a) < filter_value else 0 for a in fourier]
    n = len(fourier)
    freq = fs*np.linspace(0, n/2-1, n//2)

    plt.title(f"Transformata Fourier filtrata")
    plt.plot(freq, np.abs(fourier[:n//2]))
    plt.yscale("log")
    plt.savefig("Plot_i.pdf")


gi(18288, 1/3600, 1024, 100)
