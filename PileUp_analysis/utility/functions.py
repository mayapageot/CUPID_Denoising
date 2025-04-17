import numpy as np
import torch


def modified_gaussian(x, A, x0, sigma, b):
    """
    Compute a modified Gaussian function with a logarithmic transformation for both positive and negative x-values.

    For positive x-values, the Gaussian is computed as:
    A * exp(-(log(x) - x0)^2 / (2 * sigma^2)) + b

    For negative x-values, the Gaussian is computed as:
    A * exp(-(log(-x) - x0)^2 / (2 * sigma^2)) + b

    Parameters
    ----------
    x : array-like
        The input values for which the Gaussian is computed.
    A : float
        The amplitude of the Gaussian.
    x0 : float
        The center of the Gaussian in the log-transformed space.
    sigma : float
        The standard deviation of the Gaussian in the log-transformed space.
    b : float
        The baseline offset added to the Gaussian.

    Returns
    -------
    y : np.ndarray
        The output values corresponding to the modified Gaussian at each point in x.
    """
    # Initialize the output array with ones
    y = np.ones_like(x)

    # For x > 0, calculate the modified Gaussian using the positive side
    y[x > 0] = A * np.exp(-(np.log(x[x > 0]) - x0) ** 2 / (2 * sigma ** 2)) + b

    # For x < 0, calculate the modified Gaussian using the negative side
    y[x < 0] = A * np.exp(-(np.log(-x[x < 0]) - x0) ** 2 / (2 * sigma ** 2)) + b

    return y


def polyfit_batch(x, y, degree):
    """
    Perform polynomial fitting for each entry in a batch of x and y data.
    Fits a polynomial of the specified degree to each (x, y) pair using the least squares method.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        The x-values for each data point in the batch. Shape: (batch_size, n).
    y : np.ndarray or torch.Tensor
        The y-values for each data point in the batch. Shape: (batch_size, n).
    degree : int
        The degree of the polynomial to fit to the data.

    Returns
    -------
    coeffs : torch.Tensor
        The polynomial coefficients for each data point. Shape: (batch_size, degree + 1).
    """
    # Convert x and y to torch tensors if they are numpy arrays
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    # Create the design matrix X for polynomial fitting
    # X has shape (batch_size, n, degree+1), where each column corresponds to x^i
    X = torch.stack([x ** i for i in reversed(range(degree + 1))], dim = -1)  # (batch_size, n, degree + 1)
    XT = X.transpose(1, 2)  # (batch_size, degree + 1, n)

    # Solve the normal equation (X^T * X)^(-1) * X^T * y to get the coefficients
    coeffs = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)  # (batch_size, degree + 1)

    return coeffs


def calculate_BI(rej_power: float, dt: float) -> float:
    """
    Calculate the background interference (BI) using given parameters.

    The formula used to calculate the BI is based on the activity of a source,
    the rejection power, the exposure time, and other constants. The formula is:

    BI = (correction_factor * Probability * (1 - rej_power) * Activity^2) /
        (LMO_MASS * second_in_year) * dt / milliseconds_in_second

    Parameters
    ----------
    rej_power : float
        The rejection power (fraction of rejection).
    dt : float
        The exposure time in seconds.

    Returns
    -------
    float
        The calculated background interference (BI) value in appropriate units.
    """
    # Constants used in the BI calculation
    Activity = 2.93E-03  # Activity in Bq (N/s)
    LMO_MASS = 0.28  # Mass of LMO in kg
    second_in_year = 3600 * 24 * 365  # Number of seconds in a year
    Probability = 3.41E-04  # Given probability factor
    correction_factor = 1  # Factor for corrections
    milliseconds_in_second = 1000  # Conversion factor from seconds to milliseconds

    # BI calculation formula based on provided parameters
    return (correction_factor * Probability * (1 - rej_power) * Activity * Activity /
            LMO_MASS * second_in_year * dt / milliseconds_in_second)
