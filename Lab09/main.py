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


def exp_mean(time_series, alpha):
    s = np.zeros(len(time_series))
    s[0] = time_series[0]
    for i in range(1, len(time_series)):
        s[i] = alpha * time_series[i] + (1 - alpha) * s[i-1]
    return s


def ex2a(time_series, alpha):
    s = exp_mean(time_series, alpha)

    plt.plot(time_series, label="Seria de timp")
    plt.plot(s, label=f"Mediere exponentiala (alpha = {alpha})")
    plt.legend()
    plt.savefig("Plot_ex2a.pdf")
    plt.clf()


def calculate_error(time_series, s):
    ans = 0
    for i in range(len(time_series)-1):
        ans += (s[i]-time_series[i+1])**2
    return ans


def ex2b(time_series, no_tries):
    alphas = np.linspace(0, 1, no_tries)
    answers = np.array([])

    for alpha in alphas:
        s = exp_mean(time_series, alpha)
        answers = np.append(answers, calculate_error(time_series, s))

    chosen_alpha = 0
    chosen_answer = np.min(answers)
    for i in range(len(answers)):
        if answers[i] == chosen_answer:
            chosen_alpha = alphas[i]

    chosen_exp_mean = exp_mean(time_series, chosen_alpha)

    plt.plot(time_series, label="Seria de timp")
    plt.plot(
        s, label=f"Mediere exponentiala (alpha = {chosen_alpha})", linestyle="dotted")
    plt.legend()
    plt.savefig("Plot_ex2b.pdf")
    plt.clf()


def ex3(time_series, q):
    epsilon = np.random.normal(0, 1, len(time_series))
    theta = np.random.normal(0, 1, q)
    ma_series = np.zeros(len(time_series))
    mu = np.mean(time_series)

    for i in range(q, len(time_series)):
        ma_series[i] = epsilon[i] + np.dot(theta, epsilon[i-q:i]) + mu

    plt.plot(time_series, label="Seria de timp")
    plt.plot(ma_series, label=f"Moving average")
    plt.legend()
    plt.savefig("Plot_ex3.pdf")
    plt.clf()


time_series = ex1(1000)
ex2a(time_series, 0.2)
ex2b(time_series, 1000)
ex3(time_series, 5)
