import numpy as np
import matplotlib.pyplot as plt
import scipy


def ex4_b(signal, ws):
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label="Semnal brut")
    for w in ws:
        plt.plot(np.convolve(signal, np.ones(w), "valid")/w, label=f"w = {w}")
    plt.legend()
    plt.savefig("Plot_ex4_b.pdf")
    plt.clf()


def ex4_c(sample_freq):
    freq = sample_freq / 6  # filtram frecventele mai mari de 6 ore
    nyquist_freq = sample_freq / 2
    normalized_freq = freq / nyquist_freq
    print(f"Frecventa in Hz: {freq}")
    print(f"Frecventa normalizata: {normalized_freq}")
    return normalized_freq


def ex4_de(signal, order, rp, Wn):
    # Subpunctul d
    b_butter, a_butter = scipy.signal.butter(order, Wn)
    b_cheby, a_cheby = scipy.signal.cheby1(order, rp, Wn)

    # Subpunctul e
    signal_butter = scipy.signal.filtfilt(b_butter, a_butter, signal)
    signal_cheby = scipy.signal.filtfilt(b_cheby, a_cheby, signal)

    plt.plot(signal, label="Semnal brut")
    plt.plot(signal_butter, label="Butterworth")
    plt.plot(signal_cheby, label="Chebyshev")
    plt.legend()
    plt.savefig("Plot_ex4_e.pdf")
    plt.clf()


def ex4_f(signal, orders, rps, Wn):
    fig, axs = plt.subplots(len(orders), figsize=(12, 4 * len(orders)))
    for i in range(len(orders)):
        order = orders[i]

        axs[i].plot(signal, label="Semnal brut")

        b_butter, a_butter = scipy.signal.butter(order, Wn)
        signal_butter = scipy.signal.filtfilt(b_butter, a_butter, signal)
        axs[i].plot(signal_butter, label=f"Butterworth de ordin {order}")

        for rp in rps:
            b_cheby, a_cheby = scipy.signal.cheby1(order, rp, Wn)
            signal_cheby = scipy.signal.filtfilt(b_cheby, a_cheby, signal)
            axs[i].plot(
                signal_cheby, label=f"Chebyshev de ordin {order}, rp={rp}")

        axs[i].legend()

    plt.savefig("Plot_ex4_f.pdf")
    plt.clf()


# Subpunctul a
data = np.genfromtxt('Train.csv', delimiter=',')
signal = [int(data[i][2]) for i in range(1, len(data))][1024: (1024 + 24*3)]

ex4_b(signal, [5, 9, 13, 17])
Wn = ex4_c(1/3600)
ex4_de(signal, 5, 5, Wn)
# Mi se pare ca filtrul Butterworth filtreaza mai bine
ex4_f(signal, [1, 5, 10, 15], [5, 7, 10], Wn)
# Cred ca valorile optime pentru filtrarea frecventelor inalte sunt ordinul 10 si orice rp dintre 5, 7, 10 pentru Chebyshev
