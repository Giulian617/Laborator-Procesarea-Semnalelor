import numpy as np
import matplotlib.pyplot as plt


def generate_polynomial(n, x):
    max_power = np.random.randint(1, n+1)
    coefficients = np.random.randint(-x, x, max_power)
    return np.polynomial.Polynomial(coefficients)


def direct_convolution(p, q):
    r = np.zeros(p.degree() + q.degree() + 1)
    for i in range(p.degree() + 1):
        for j in range(q.degree() + 1):
            r[i+j] += p.coef[i] * q.coef[j]
    return np.polynomial.Polynomial(r)


def fft_convolution(p, q):
    p_fft = np.fft.fft(p.coef, p.degree() + q.degree() + 1)
    q_fft = np.fft.fft(q.coef, p.degree() + q.degree() + 1)
    return np.polynomial.Polynomial(np.fft.ifft(p_fft*q_fft).real)


def ex2(n, x):
    p = generate_polynomial(n, x)
    q = generate_polynomial(n, x)

    print(f"p: {p}")
    print(f"q: {q}")
    print(direct_convolution(p, q))
    print(fft_convolution(p, q))


ex2(10, 30)
