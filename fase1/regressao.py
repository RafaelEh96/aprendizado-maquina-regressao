import numpy as np

def regressao(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    beta1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    beta0 = y_mean - beta1 * x_mean

    return beta0, beta1