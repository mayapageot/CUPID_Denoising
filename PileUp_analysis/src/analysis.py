import numpy as np
from scipy.signal.windows import tukey
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.relpath("../utility"))
from functions import polyfit_batch

def compute_transfer_function(fs, points, position, meanpulse, bslspec):
    """
    Compute a frequency-domain transfer function based on a mean pulse and baseline spectrum.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    points : int
        Number of points in the FFT (window size).
    position : float
        Relative position (0–1) where the pulse is assumed to be centered.
    meanpulse : np.ndarray
        Mean pulse shape.
    bslspec : np.ndarray
        Baseline spectral density.

    Returns
    -------
    transferfunc : np.ndarray
        Frequency-domain transfer function.
    func_pulse_TF : np.ndarray
        Filtered pulse in the time domain.
    """
    # Compute difference between mean values
    meanpulse_dif = np.diff(np.append(meanpulse, meanpulse[-1]))

    # Create frequency array for FFT
    freq = np.linspace(0, fs - fs / points, points)

    # Apply Tukey window
    win = tukey(points)

    # Half data length (used later for slicing)
    hd = points // 2

    # Compute normalization factor k
    fft_mean_win = np.fft.fft(meanpulse_dif * win)
    k = points / np.sum(np.abs(fft_mean_win) ** 2 / bslspec)

    # Frequency domain transfer function
    shift = np.exp(-1j * 2 * np.pi * freq * (np.ceil(position * points) / fs))
    transf = k * (np.conj(np.fft.fft(meanpulse_dif)) / bslspec) * shift

    # Convert transfer function to time domain
    itransf = np.real(np.fft.ifft(transf))

    # Window and pad the impulse response
    itransf_conc = np.concatenate([
        itransf[:hd] * win[hd:],         # First half with window
        np.zeros(points),               # Zero padding
        itransf[hd:] * win[:hd]         # Second half with window
    ])

    # Compute final FFT of time-domain impulse response
    transferfunc = np.fft.fft(itransf_conc)

    # Prepare a padded and windowed pulse for filtering
    pulse = np.concatenate([
        np.zeros(points // 2),
        meanpulse_dif * win,
        np.zeros(points // 2)
    ])
    PULSE = np.fft.fft(pulse)

    # Apply transfer function in frequency domain
    FPULSE = transferfunc * PULSE

    # Convert filtered pulse back to time domain
    func_pulse_TF = np.real(np.fft.ifft(FPULSE))

    return transferfunc, func_pulse_TF
def get_filtered_pulse(dpulses, transferfunc, windowing=tukey):
    """
    Filter a derivative pulse (or pulses) using a frequency-domain transfer function.

    Parameters
    ----------
    dpulses : np.ndarray
        Input derivative pulses, shape (n,) or (batch_size, n).
    transferfunc : np.ndarray
        Frequency-domain transfer function.
    windowing : callable, optional
        Windowing function (default: scipy's tukey).

    Returns
    -------
    fpulse : np.ndarray
        Filtered pulses in the time domain, same shape as input.
    """

    t_size = len(transferfunc)

    if dpulses.ndim == 1:  # Single pulse
        win_size = len(dpulses)
        win = windowing(win_size)
        if t_size > win_size:
            # Pad and apply FFT
            f_pulse = np.fft.fft(np.concatenate([np.zeros(win_size // 2), dpulses, np.zeros(win_size // 2)]))
            fpulse = np.real(np.fft.ifft(f_pulse * transferfunc))
        else:
            f_pulse = np.fft.fft(dpulses)
            fpulse = np.real(np.fft.ifft(f_pulse * transferfunc))
    else:  # Multiple pulses (batch)
        n_p, win_size = np.shape(dpulses)
        win = windowing(win_size)
        if t_size > win_size:
            # Pad batch of pulses
            dpulses = np.concatenate([np.zeros((n_p, win_size // 2)), dpulses, np.zeros((n_p, win_size // 2))], axis=1)
        f_pulse = np.fft.fft(dpulses, axis=1)
        fpulse = np.real(np.fft.ifft(f_pulse * transferfunc, axis=1))

    return fpulse


def get_interpolate_filtered_amp(dataset, transferfunc, arg_max=1999):
    """
    Interpolate around the maximum of a filtered pulse to find sub-sample peak positions.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset of pulses.
    transferfunc : np.ndarray
        Frequency-domain transfer function.
    arg_max : int, optional
        Approximate peak index in the filtered pulse (default: 1999).

    Returns
    -------
    peak_values : np.ndarray
        Interpolated peak values.
    peak_positions : np.ndarray
        Sub-sample positions of the peak.
    """

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for pulses in loader:
        if hasattr(pulses, 'numpy'):
            pulses = pulses.numpy()

        n_p, win_size = np.shape(pulses)
        row_indices = np.arange(n_p)[:, None]

        # Compute first derivative (difference)
        dpulses = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        fpulses = get_filtered_pulse(dpulses, transferfunc)

        # Window of points around the peak to interpolate
        x = np.arange(-5, 6) + arg_max
        y = fpulses[:, arg_max - 5: arg_max + 6]

        # Interpolation function for sub-sample peak finding
        interp_func = interp1d(x, y, kind='cubic', axis=1, bounds_error=False, fill_value='extrapolate')

        x_inter = np.arange(-1, 1, 0.01) + arg_max
        inter_fpulse = interp_func(x_inter)

        arg_max_inter = np.argmax(inter_fpulse, axis=1)

        return (
            inter_fpulse[row_indices, arg_max_inter],  # peak values
            x_inter[arg_max_inter]                    # sub-sample peak positions
        )


def get_interpolate_fitted_amp(dataset, transferfunc, arg_max, meanpulse, jitter=150):
    """
    Estimate the amplitude of pulses by fitting the filtered mean to each filtered pulse.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset of pulses.
    transferfunc : np.ndarray
        Frequency-domain transfer function.
    arg_max : np.ndarray
        Array of sub-sample peak positions per pulse.
    meanpulse : np.ndarray
        Average pulse shape used for fitting.
    jitter : int, optional
        Time window size around the peak for fitting (default: 150).

    Returns
    -------
    amplitudes : np.ndarray
        Estimated amplitude for each pulse.
    """

    # Prepare filtered mean pulse
    mean_dif = np.diff(np.append(meanpulse, meanpulse[-1]))
    fmeanpulse = get_filtered_pulse(mean_dif, transferfunc)

    x = np.arange(len(fmeanpulse))
    y = fmeanpulse
    interp_func_mean = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for pulses in loader:
        if hasattr(pulses, 'numpy'):
            pulses = pulses.numpy()
        n_p, win_size = np.shape(pulses)
        row_indices = np.arange(n_p)[:, None]

        # Compute filtered pulses
        dpulses = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        fpulses = get_filtered_pulse(dpulses, transferfunc)

        # Fit linear amplitude to the filtered mean within a jitter window
        jitter_array = np.arange(-jitter, jitter + 1)
        coeff = polyfit_batch(
            interp_func_mean(jitter_array[None, :] + arg_max[:, None]),
            fpulses[:, 1999 - jitter:1999 + jitter + 1],
            1
        )

        return coeff[:, 0]  # return amplitude coefficients


def get_interpolate_chi2(dataset, transferfunc, arg_max, meanpulse):
    """
    Compute chi-squared distances between filtered pulses and a reference pulse
    shifted in time, in the frequency domain.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset of pulses.
    transferfunc : np.ndarray
        Frequency-domain transfer function.
    arg_max : np.ndarray
        Sub-sample peak positions to align reference pulse.
    meanpulse : np.ndarray
        Mean pulse used as reference.

    Returns
    -------
    chi2 : np.ndarray
        Chi-squared distances for each pulse.
    """

    # Prepare frequency-domain normalized reference signal
    mean_dif = np.diff(np.append(meanpulse, meanpulse[-1]))
    f_meanpulse = np.fft.fft(np.concatenate([np.zeros(1000), mean_dif, np.zeros(1000)])) * transferfunc
    f_meanpulse = f_meanpulse / np.linalg.norm(f_meanpulse)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    freq = np.fft.fftfreq(4000, 1e-4)
    dt = arg_max - 1999

    for pulses in loader:
        # Compute FFT of the pulse derivatives
        diff_pulse = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        f_pulse = np.fft.fft(diff_pulse, axis=1) * transferfunc
        f_pulse = f_pulse / np.linalg.norm(f_pulse, axis=1)[:, None]

        # Chi-square distance between phase-shifted pulses and reference
        chi2 = np.sum(
            np.abs(
                f_pulse * np.exp(-1j * 2 * np.pi * freq * dt[:, None]) - f_meanpulse[None, :]
            ) ** 2,
            axis=1
        )

    return chi2

def compute_transfer_simple(mean, ANPS):
    """
    Convenience wrapper to compute a transfer function and filtered mean using preset parameters.

    Parameters
    ----------
    mean : np.ndarray
        Mean pulse waveform.
    ANPS : np.ndarray
        Baseline noise power spectral density.

    Returns
    -------
    transferfunc : np.ndarray
        Frequency-domain transfer function.
    func_pulse_TF : np.ndarray
        Time-domain filtered mean pulse.
    """
    # A simple wrapper around compute_transfer_function
    t1, t1_small = compute_transfer_function(10000, 2000, 0.5, mean, ANPS)
    return t1, t1_small