import numpy as np
import matplotlib.pyplot as plt


def ex1_a(n):
    x = np.zeros((n, n))
    for n1 in range(n):
        for n2 in range(n):
            x[n1][n2] = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x, cmap=plt.cm.gray)
    axs[0].set_title("Functia")

    axs[1].imshow(20 * np.log10(abs(np.fft.fft2(x)) + 1e-15), cmap=plt.cm.gray)
    axs[1].set_title("Spectrul functiei")

    plt.savefig("Plot_ex1_a.pdf")
    plt.clf()


def ex1_b(n):
    x = np.zeros((n, n))
    for n1 in range(n):
        for n2 in range(n):
            x[n1][n2] = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x, cmap=plt.cm.gray)
    axs[0].set_title("Functia")

    axs[1].imshow(20 * np.log10(abs(np.fft.fft2(x)) + 1e-15), cmap=plt.cm.gray)
    axs[1].set_title("Spectrul functiei")

    plt.savefig("Plot_ex1_b.pdf")
    plt.clf()


def ex1_cde(Y, idx):
    x = np.real(np.fft.ifft2(Y))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x, cmap=plt.cm.gray)
    axs[0].set_title("Functia")

    axs[1].imshow(Y, cmap=plt.cm.gray)
    axs[1].set_title("Spectrul functiei")

    plt.savefig(f"Plot_ex1_{idx}.pdf")
    plt.clf()


def ex1_aux(n, i1, j1, i2, j2, idx):
    Y = np.zeros((n, n))
    Y[i1][j1] = 1
    Y[i2][j2] = 1
    ex1_cde(Y, idx)


n = 128
ex1_a(n)
ex1_b(n)
ex1_aux(n, 0, 5, 0, n-5, "c")
ex1_aux(n, 5, 0, n-5, 0, "d")
ex1_aux(n, 5, 5, n-5, n-5, "e")
