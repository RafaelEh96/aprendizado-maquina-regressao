import numpy as np
import matplotlib.pyplot as plt

def correlacao(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerador   = np.sum((x - x_mean) * (y - y_mean))
    denominador = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    if denominador == 0:
        raise ValueError("Denominador zero: variância nula em x ou y.")

    r = numerador / denominador
    return r
