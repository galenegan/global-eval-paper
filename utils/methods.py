"""Error metrics and Extended Triple Collocation used by the plotting and table
scripts in this repository."""

import numpy as np


def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true), axis=0)


def bias(y_pred, y_true):
    return np.mean((y_pred - y_true), axis=0)


def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))


def ETC(y: np.ndarray):
    """
    Extended Triple Collocation (ETC)

    Parameters
    ----------
    y : (N, 3) ndarray
        Temporally/spatially collocated observations from three systems.
        No NaNs allowed.

    Returns
    -------
    errVar_ETC : (3,) ndarray
        Estimated noise variances for each system w.r.t. the unknown truth.
        Index i corresponds to column i of `y`.

    rho2 : (3,) ndarray
        Squared correlation coefficients of each system w.r.t. the unknown truth.

    beta : (3,) ndarray
        Estimated slopes relative to the reference x = y[:, 0]

    alpha : (3,) ndarray
        Estimated biases relative to the reference x = y[:, 0]

    Reference
    ---------
    McColl, K.A., Vogelzang, J., Konings, A.G., Entekhabi, D., Piles, M., & Stoffelen, A. (2014).
    Extended Triple Collocation: Estimating errors and correlation coefficients with respect to an
    unknown target. Geophysical Research Letters, 41, 6229-6236.
    """
    # ---- Input checks ----
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("Input data y must be an N x 3 matrix")
    if np.isnan(y).any():
        raise ValueError("Input data y must not contain NaNs")

    # sample variance of each column must be non-zero
    if any(np.all(y[:, i] == y[0, i]) for i in range(3)):
        raise ValueError(
            "The sample variance of each of the columns of y must be non-zero. "
            "Increase your sample size or reconsider using ETC."
        )

    # ---- Covariance matrix (columns are variables, ddof=1) ----
    Q_hat = np.cov(y, rowvar=False, ddof=1)

    # ---- Error variances ----
    err1 = Q_hat[0, 0] - Q_hat[0, 1] * Q_hat[0, 2] / Q_hat[1, 2]
    err2 = Q_hat[1, 1] - Q_hat[0, 1] * Q_hat[1, 2] / Q_hat[0, 2]
    err3 = Q_hat[2, 2] - Q_hat[0, 2] * Q_hat[1, 2] / Q_hat[0, 1]
    err_var = np.array([err1, err2, err3], dtype=float)

    # ---- Correlation coefficients ----
    r1 = np.sqrt(Q_hat[0, 1] * Q_hat[0, 2] / Q_hat[0, 0] / Q_hat[1, 2])
    r2 = np.sign(Q_hat[0, 2] * Q_hat[1, 2]) * np.sqrt(Q_hat[0, 1] * Q_hat[1, 2] / Q_hat[1, 1] / Q_hat[0, 2])
    r3 = np.sign(Q_hat[0, 1] * Q_hat[1, 2]) * np.sqrt(Q_hat[0, 2] * Q_hat[1, 2] / Q_hat[2, 2] / Q_hat[0, 1])
    rho = np.array([r1, r2, r3], dtype=float)
    rho2 = rho ** 2

    # ---- Affine coefficients relative to the reference (column 0) ----
    beta_1 = 1
    alpha_1 = 0
    beta_2 = Q_hat[1, 2] / Q_hat[0, 2]
    alpha_2 = np.mean(y[:, 1]) - beta_2 * np.mean(y[:, 0])
    beta_3 = Q_hat[1, 2] / Q_hat[0, 1]
    alpha_3 = np.mean(y[:, 2]) - beta_3 * np.mean(y[:, 0])
    beta = np.array([beta_1, beta_2, beta_3], dtype=float)
    alpha = np.array([alpha_1, alpha_2, alpha_3], dtype=float)

    return err_var, rho2, beta, alpha
