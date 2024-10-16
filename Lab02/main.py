import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice


def ex1(freq, no_samples):
    fig, axs = plt.subplots(2)
    fig.suptitle("Sinus si cosinus")

    samples = np.linspace(0, 1, no_samples)
    axs[0].plot(samples, np.sin(2 * np.pi * freq * samples))
    axs[1].plot(samples, np.cos(2 * np.pi * freq * samples + np.pi/2))

    plt.savefig("Plot_ex1.pdf")
    plt.clf()


def ex2_a(freq, no_samples, phases):
    fig, axs = plt.subplots(4)
    fig.suptitle("Sinusoide defazate")

    samples = np.linspace(0, 1, no_samples)

    for i in range(4):
        axs[i].plot(samples, np.sin(2 * np.pi * freq * samples + phases[i]))

    plt.savefig("Plot_ex2_a.pdf")
    plt.clf()


def ex2_b(freq, no_samples, z, snr):
    fig, axs = plt.subplots(5)
    fig.suptitle("Sinusoide cu zgomot")

    samples = np.linspace(0, 1, no_samples)
    x_mean = np.linalg.norm(samples)
    z_mean = np.linalg.norm(z)

    for i in range(0, 5):
        if snr[i] == 0:
            gama = 0
        else:
            gama = x_mean / (z_mean * np.sqrt(snr[i]))
            # gama = np.sqrt(x_mean**2 / (snr[i] * (z_mean**2)))
        axs[i].plot(samples, np.sin(2 * np.pi * freq * samples) + gama * z)

    plt.savefig("Plot_ex2_b.pdf")
    plt.clf()


def playSignal(signal, rate):
    sounddevice.play(signal, rate)
    sounddevice.wait()


def ex3():
    # ex2_a:
    samples = np.linspace(0, 1, int(1e5))
    #playSignal(np.sin(2 * np.pi * 400 * samples), 44100)

    # ex2_b
    samples = np.linspace(0, 1, int(1e5))
    #playSignal(np.sin(2 * np.pi * 800 * samples), 44100)

    # ex2_c
    samples = np.linspace(0, 1, int(1e5))
    #playSignal(np.mod(samples, 1), 44100)

    # ex2_d
    samples = np.linspace(0, 1, int(1e5))
    #playSignal(np.sign(np.sin(2 * np.pi * 300 * samples)), 44100)

    scipy.io.wavfile.write("ex2_d.wav", int(
        1e5), np.sign(np.sin(2 * np.pi * 300 * samples)))

    rate, x = scipy.io.wavfile.read("ex2_d.wav")
    #playSignal(x, rate)


def ex4(freq, no_samples):
    fig, axs = plt.subplots(3)
    fig.suptitle("Sinusoida + Sawtooth")

    samples = np.linspace(0, 5, no_samples)
    axs[0].plot(samples, np.sin(2 * np.pi * freq * samples))
    axs[1].plot(samples, np.mod(samples, 1))
    axs[2].plot(samples, np.sin(
        2 * np.pi * freq * samples) + np.mod(samples, 1))

    plt.savefig("Plot_ex4.pdf")
    plt.clf()


def ex5(freq1, freq2, no_samples):
    samples = np.linspace(0, 1, no_samples)
    playSignal(np.concatenate((np.sin(2 * np.pi * freq1 * samples), np.sin(2 * np.pi * freq2 * samples))), 44100)
    # nu-mi dau seama exact care e diferenta dintre ele, dar primul mi se pare in registru grav, iar celalalt in registru acut =))

def ex6(no_samples):
    f = np.array([no_samples/2, no_samples/4, 0])

    fig, axs = plt.subplots(3)
    fig.suptitle("Sinusoide cu frecventa diferita")

    samples = np.linspace(0, 1, no_samples)
    for i in range(3):
        axs[i].plot(samples, np.sin(2 * np.pi * f[i] * samples))

    plt.savefig("Plot_ex6.pdf")
    plt.clf()


def ex7(freq, no_samples):
    fig, axs = plt.subplots(3)
    fig.suptitle("Sinusoide decimate")

    samples = np.linspace(0, 10, no_samples)
    samples_v1 = []
    for i in range(0, no_samples, 4):
        samples_v1.append(samples[i]/4)
    samples_v2 = []
    for i in range(1, len(samples_v1), 4):
        samples_v2.append(samples_v1[i]/4)
    samples_v1 = np.array(samples_v1)
    samples_v2 = np.array(samples_v2)

    axs[0].plot(samples, np.sin(2 * np.pi * freq * samples))
    axs[1].plot(samples_v1, np.sin(2 * np.pi * freq * samples_v1))
    axs[2].plot(samples_v2, np.sin(2 * np.pi * freq * samples_v2))

    plt.savefig("Plot_ex7.pdf")
    plt.clf()

    # Frecventa se imparte la 4 de fiecare data, dar forma de sinusoida se pastreaza


def ex8(no_samples):
    fig, axs = plt.subplots(5)
    fig.suptitle("Aproximare sinus")

    samples = np.linspace(-np.pi / 2, np.pi / 2, no_samples)
    axs[0].plot(samples, np.sin(samples))
    axs[1].plot(samples, samples)
    axs[2].plot(samples, (samples - 7 * (samples**3) / 60) /
                (1 + (samples**2) / 20))
    axs[3].plot(samples, np.abs(samples - np.sin(samples)))
    axs[4].plot(samples, np.abs((samples - 7 * (samples**3) / 60) /
                (1 + (samples**2) / 20) - np.sin(samples)))

    axs[3].set_yscale("log")
    axs[4].set_yscale("log")

    plt.savefig("Plot_ex8.pdf")
    plt.clf()


ex1(100, 200)
ex2_a(100, 200, [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
ex2_b(10, 500, np.random.normal(0, 1, 500), [0, 0.1, 1, 10, 100])
ex3()
ex4(100, 200)
ex5(200, 500, int(1e5))
ex6(200)
ex7(400, 1000)
ex8(1000)
