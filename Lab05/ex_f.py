import numpy as np


def f(n, fs):
    data = np.genfromtxt('Train.csv', delimiter=',')
    x = [int(data[i][2]) for i in range(1, len(data))]
    x -= np.mean(x)

    fourier = np.fft.fft(x)/np.sqrt(n)
    max_4freq = sorted(
        zip(np.abs(fourier[:n//2]), np.linspace(0, n/2-1, n//2)/24))[-4:]

    print(*max_4freq, sep="\n")


f(18288, 1/3600)
