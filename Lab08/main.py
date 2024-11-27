import numpy as np
import matplotlib.pyplot as plt


def ex1(n):
    times = np.linspace(0, n, n)
    trend = 1 + 2 * times + times ** 2
    trend = trend / np.max(trend)
    season = 3 * np.sin(2 * np.pi * 7 * times) + 11 * \
        np.sin(2 * np.pi * 3 * times)
    residuals = np.random.normal(0, 1, n)
    time_series = trend + season + residuals
    time_Series = time_series / np.max(time_series)

    fig, axs = plt.subplots(4)

    axs[0].set_title("Time series")
    axs[0].plot(time_series)

    axs[1].set_title("Trend")
    axs[1].plot(trend)

    axs[2].set_title("Season")
    axs[2].plot(season)

    axs[3].set_title("Residuals")
    axs[3].plot(residuals)

    plt.tight_layout()
    plt.savefig("Plot_ex1.pdf")
    plt.clf()

    return time_series


def ex2(time_series):
    def autocorrelate(time_series, idx):
        length = len(time_series)
        time_series -= np.mean(time_series)
        return np.sum(time_series[:length-idx] * time_series[idx:]) / (length * np.var(time_series))

    ans = [autocorrelate(time_series, i) for i in range(len(time_series))]
    ans /= np.max(ans)

    np_ans = np.correlate(time_series, time_series, mode="full")
    np_ans = np_ans[len(np_ans)//2:]
    np_ans /= np.max(np_ans)

    fig, axs = plt.subplots(2)

    axs[0].set_title("Manual")
    axs[0].plot(ans)

    axs[1].set_title("Using numpy")
    axs[1].plot(np_ans)

    plt.tight_layout()
    plt.savefig("Plot_ex2.pdf")
    plt.clf()


time_series = ex1(1000)
ex2(time_series)
