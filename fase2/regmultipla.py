import numpy as np

def regmultipla(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = X.shape[0]

    ones = np.ones((m, 1))
    X_bias = np.hstack([ones, X])

    Xt = X_bias.T
    beta = np.linalg.inv(Xt @ X_bias) @ Xt @ y

    return beta