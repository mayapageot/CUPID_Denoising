import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
def compute_mahalanobis_distance_squared(x, y, mean, cov):
    """
    Compute squared Mahalanobis distance from (x, y) to a distribution with given mean and covariance.

    Returns:
    - d2: array of squared distances
    """
    data = np.column_stack((x, y))
    diffs = data - mean
    inv_cov = np.linalg.inv(cov)
    d2 = np.einsum('ij,jk,ik->i', diffs, inv_cov, diffs)
    return d2
from scipy.stats import chi2

def get_inliers_outliers(x, y, mean, cov, confidence=0.90, ):
    """
    Returns inliers and outliers based on Mahalanobis distance and a confidence level.

    Returns:
    - inliers: Nx2 array of inlier points
    - outliers: Nx2 array of outlier points
    """
    d2 = compute_mahalanobis_distance_squared(x, y, mean, cov)
    threshold = chi2.ppf(confidence, df=2)
    mask = d2 <= threshold
    data = np.column_stack((x, y))
    return data[mask], data[~mask]

def get_mask(x, y, mean, cov, confidence=0.90, ):
    """
    Returns inliers and outliers based on Mahalanobis distance and a confidence level.

    Returns:
    - inliers: Nx2 array of inlier points
    - outliers: Nx2 array of outlier points
    """
    d2 = compute_mahalanobis_distance_squared(x, y, mean, cov)
    threshold = chi2.ppf(confidence, df=2)
    mask = d2 <= threshold
    return mask

def draw_confidence_ellipse(mean, cov, confidence=0.90, ax=None, **kwargs):
    """
    Draw a confidence ellipse given mean and covariance.

    Parameters:
    - mean: (2,) array
    - cov: (2,2) array
    - confidence: Confidence level for ellipse
    - ax: Optional matplotlib axis
    - kwargs: Passed to Ellipse (e.g., color, linestyle)
    """
    from scipy.stats import chi2

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    chi2_val = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    if ax is None:
        fig, ax = plt.subplots()
    elipse_kwargs = dict(edgecolor='blue', fc='None', lw=2)
    elipse_kwargs.update(kwargs)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **elipse_kwargs)
    ax.add_patch(ellipse)

    return ax


def asymmetric_rotated_ellipse_cut(x, y, x0, y0, a_left, a_right, b, theta):
    # Shift to center
    dx = x - x0
    dy = y - y0

    # Rotate (clockwise to align ellipse frame)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    # Asymmetric semi-axis
    a = np.where(x_rot < 0, a_left, a_right)

    # Ellipse condition
    inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 < 1
    return inside


def find_best_asymmetric_ellipse(signal, background, eff_target=0.9):
    x_sig, y_sig = signal[:, 0], signal[:, 1]
    x_bkg, y_bkg = background[:, 0], background[:, 1]

    x0, y0 = np.median(x_sig), np.median(y_sig)
    cov = np.cov(signal, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    chi2_val = chi2.ppf(eff_target, df=2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    theta = np.arctan2(*eigvecs[:, 0][::-1])

    a_vals = np.linspace(width * 0.2, width, 40)
    b_vals = np.linspace(height * 0.2, height, 40)

    best_params = None
    best_bkg_count = np.inf

    for a_left in a_vals:
        for a_right in a_vals:
            for b in b_vals:
                sig_mask = asymmetric_rotated_ellipse_cut(x_sig, y_sig, x0, y0, a_left, a_right, b, theta)
                bkg_mask = asymmetric_rotated_ellipse_cut(x_bkg, y_bkg, x0, y0, a_left, a_right, b, theta)

                sig_eff = np.sum(sig_mask) / len(x_sig)
                bkg_count = np.sum(bkg_mask)

                if sig_eff >= eff_target and bkg_count < best_bkg_count:
                    best_bkg_count = bkg_count
                    best_params = (x0, y0, a_left, a_right, b)

    return best_params, best_bkg_count, theta


def plot_asymmetric_rotated_ellipse(x0, y0, a_left, a_right, b, theta, ax=None, **kwargs):
    """
    Plot the asymmetric rotated ellipse boundary.

    Parameters:
        x0, y0       : Center
        a_left/right : Semi-axis (left or right in rotated frame)
        b            : Vertical semi-axis
        theta        : Rotation (radians)
        ax           : Matplotlib axis
        **kwargs     : Additional args to matplotlib.plot()
    """
    # Create angles for upper half
    phi = np.linspace(0, 2 * np.pi, 500)

    # Base ellipse in its own frame
    x_unit = np.cos(phi)
    y_unit = np.sin(phi)

    # Asymmetric scaling on x depending on side
    a = np.where(x_unit < 0, a_left, a_right)
    x_scaled = a * x_unit
    y_scaled = b * y_unit

    # Rotate back to global coords
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = x_scaled * cos_t - y_scaled * sin_t
    y_rot = x_scaled * sin_t + y_scaled * cos_t

    # Shift to center
    x_final = x0 + x_rot
    y_final = y0 + y_rot

    if ax is None:
        ax = plt.gca()
    ax.plot(x_final, y_final, **kwargs)
