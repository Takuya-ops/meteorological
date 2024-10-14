import numpy as np
import matplotlib.pyplot as plt


def fourier_square_wave(x, n_terms):
    result = 0
    for n in range(1, n_terms + 1, 2):
        result += (4 / (n * np.pi)) * np.sin(n * x)
    return result


x = np.linspace(0, 4 * np.pi, 1000)

plt.figure(figsize=(12, 8))

# 元の矩形波
plt.plot(
    x, np.where((x % (2 * np.pi)) < np.pi, 1, 0), "k--", label="Original Square Wave"
)

# フーリエ級数による近似
for n_terms in [1, 3, 5, 20]:
    y = fourier_square_wave(x, n_terms)
    plt.plot(x, y, label=f"Fourier ({n_terms} terms)")

plt.legend()
plt.title("Square Wave Approximation using Fourier Series")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
