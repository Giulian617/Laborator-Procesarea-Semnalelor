import numpy as np
import matplotlib.pyplot as plt


def ex1_a(start, end, freq):
    return np.linspace(start, end, freq)


def ex1_b(show_points, end):
    fig, axs = plt.subplots(3)
    fig.suptitle("Semnale continue")
    axs[0].plot(show_points, np.cos(520 * np.pi * show_points + np.pi / 3))
    axs[1].plot(show_points, np.cos(280 * np.pi * show_points - np.pi / 3))
    axs[2].plot(show_points, np.cos(120 * np.pi * show_points + np.pi / 3))

    axs[0].set_ylabel("x(t)")
    axs[1].set_ylabel("y(t)")
    axs[2].set_ylabel("z(t)")

    for ax in axs:
        ax.set_xlim(0, end)

    plt.savefig("Plot_ex1_a_b.pdf")
    plt.clf()


def ex1_c(show_points, end):
    fig, axs = plt.subplots(3)
    fig.suptitle("Semnale esantionate")
    axs[0].stem(show_points, np.cos(520 * np.pi * show_points + np.pi / 3))
    axs[1].stem(show_points, np.cos(280 * np.pi * show_points - np.pi / 3))
    axs[2].stem(show_points, np.cos(120 * np.pi * show_points + np.pi / 3))

    axs[0].set_ylabel("x[n]")
    axs[1].set_ylabel("y[n]")
    axs[2].set_ylabel("z[n]")

    for ax in axs:
        ax.set_xlim(0, end)

    plt.savefig("Plot_ex1_c.pdf")
    plt.clf()


def ex2_a(freq, no_samples):
    samples = np.linspace(0, 0.2, no_samples)
    # am lasat pe plot pt ca arata mai bine
    plt.plot(samples, np.sin(2 * np.pi * freq * samples))

    plt.title("Semnal sinusoidal de frecventa 400Hz")
    plt.xlabel("Timp")
    plt.ylabel("x[n]")

    plt.savefig("Plot_ex2_a.pdf")
    plt.cla()


def ex2_b(freq, no_samples):
    samples = np.linspace(0, 3, no_samples)
    plt.plot(samples, np.sin(2 * np.pi * freq * samples))

    plt.title("Semnal sinusoidal de frecventa 800Hz")
    plt.xlabel("Timp")
    plt.ylabel("x[n]")

    plt.savefig("Plot_ex2_b.pdf")
    plt.cla()


def ex2_c(freq, no_samples):
    samples = np.linspace(0, 5, no_samples)
    plt.plot(samples, np.mod(samples, 1))

    plt.title("Semnal sawtooth de frecventa 240Hz")
    plt.xlabel("Timp")
    plt.ylabel("x[n]")

    plt.savefig("Plot_ex2_c.pdf")
    plt.cla()


def ex2_d(freq, no_samples):
    samples = np.linspace(0, 5, no_samples)
    plt.plot(samples, np.sign(np.sin(2 * np.pi * freq * samples)))

    plt.title("Semnal square de frecventa 300Hz")
    plt.xlabel("Timp")
    plt.ylabel("x[n]")

    plt.savefig("Plot_ex2_d.pdf")
    plt.cla()


def ex2_e(dim):
    matrix = np.random.rand(dim, dim)

    plt.title("Semnal 2D aleator")
    plt.imshow(matrix)

    plt.savefig("Plot_ex2_e.pdf")
    plt.cla()


def ex2_f(dim):
    matrix = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            if (i+j) % 2 == 0:
                matrix[i][j] = 1

    plt.title("Semnal 2D alternativ")
    plt.imshow(matrix)

    plt.savefig("Plot_ex2_f.pdf")
    plt.cla()


def ex3_a(no_samples):
    print(f"Intervalul de timp dintre doua esantioane este de {1/no_samples}.")


def ex3_b(no_samples):
    print(f"O ora de achizitie va ocupa {4 * no_samples * 3600 / 8} bytes.")


ex1_b(ex1_a(0, 0.03, int(0.03/0.0005)), 0.03)
ex1_c(ex1_a(0, 0.03, 200), 0.03)
ex2_a(400, 1600)
ex2_b(800, 200)
ex2_c(240, 500)
ex2_d(300, 500)
ex2_e(128)
ex2_f(128)
ex3_a(2000)
ex3_b(2000)
