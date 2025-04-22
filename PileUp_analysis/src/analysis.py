import numpy as np
from scipy.signal.windows import tukey
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d

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


def get_interpolate_filtered_amp(dataset, transferfunc, arg_max=1999, pos_max_inter=None, subsample_size=0.01, jitter_max=4):
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
    pos_max_inter: np.ndarray, optional
        Position of the amplitude to return, if None search for the maximum around arg_max
    subsample_size : float, optional
        Subsize of the interpolation bins (default: 0.01)
    jitter_max : int, optional
        Region of search of maximum (default: 4)
    Returns
    -------
    peak_values : np.ndarray
        Interpolated peak values.
    peak_positions : np.ndarray
        Sub-sample positions of the peak.
    """
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if pos_max_inter is None :
        pos_max_inter_split = [None]*len(loader)
    else :
        pos_max_inter_split = [pos_max_inter[i:i + batch_size]for i in range(0, len(pos_max_inter), batch_size)]

    peak_values = []
    peak_positions = []
    for pulses, pos_max_inter_sub in zip(loader,pos_max_inter_split):
        if hasattr(pulses, 'numpy'):
            pulses = pulses.numpy()

        n_p, win_size = np.shape(pulses)
        row_indices = np.arange(n_p)

        # Compute first derivative (difference)
        dpulses = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        fpulses = get_filtered_pulse(dpulses, transferfunc)

        # Window of points around the peak to interpolate
        x = np.arange(-5, 6) + arg_max
        y = fpulses[:, arg_max - 5: arg_max + 6]

        # Interpolation function for sub-sample peak finding
        interp_func = interp1d(x, y, kind='cubic', axis=1, bounds_error=False, fill_value='extrapolate')

        x_inter = np.arange(-jitter_max, jitter_max+subsample_size, subsample_size) + arg_max
        inter_fpulse = interp_func(x_inter)
        arg_max_inter_sub = np.argmax(inter_fpulse, axis = 1)
        if pos_max_inter_sub is None :
            peak_values.append(inter_fpulse[row_indices, arg_max_inter_sub])

        else :
            # print(pos_max_inter_sub.shape)
            # print(pulses.shape)
            peak_values.append(inter_fpulse[pos_max_inter_sub[:,None] == x_inter])
        peak_positions.append(x_inter[arg_max_inter_sub])
    peak_values = np.concatenate(peak_values)
    peak_positions = np.concatenate(peak_positions)
    return (
        peak_values,  # peak values
        peak_positions  # sub-sample peak positions
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
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    freq = np.fft.fftfreq(4000, 1e-4)
    dt = (arg_max - 1999) * 1e-4
    dt_split = [dt[i:i + batch_size] for i in range(0, len(dt), batch_size)]
    coeffs = []
    for pulses,dt_sub in zip(loader,dt_split):
        if hasattr(pulses, 'numpy'):
            pulses = pulses.numpy()
        n_p, win_size = np.shape(pulses)
        row_indices = np.arange(n_p)[:, None]

        # Compute filtered pulses
        dpulses = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        f_pulse = np.fft.fft(dpulses, axis = 1)* np.exp(1j * 2 * np.pi * freq * dt_sub[:, None])

        fpulses = np.real(np.fft.ifft(f_pulse * transferfunc))
        # Fit linear amplitude to the filtered mean within a jitter window

        coeff = np.polyfit(
            fmeanpulse[1999 - jitter:1999 + jitter + 1],
            fpulses[:, 1999 - jitter:1999 + jitter + 1].T,
            1
        ).T
        coeffs.append(coeff[:, 0])
    coeffs = np.concatenate(coeffs)
    return coeffs  # return amplitude coefficients


def get_PSD(dataset, transferfunc, meanpulse, arg_max=1999, transferfunc_max=None, subsample_size=0.01, jitter_max=4,jitter_fit=150):
    """
    Interpolate around the maximum of a filtered pulse to find sub-sample peak positions.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset of pulses.
    transferfunc : np.ndarray
        Frequency-domain transfer function.
    meanpulse: np.ndarray
        Average pulse shape used for fitting.
    arg_max : int, optional
        Approximate peak index in the filtered pulse (default: 1999).
    transferfunc_max : np.ndarray
        Frequency-domain transfer function for maxamp finding.
    subsample_size : float, optional
        Subsize of the interpolation bins (default: 0.01)
    jitter_max : int, optional
        Region of search of maximum (default: 4)
    jitter_fit : int, optional
        Time window size around the peak for fitting (default: 150).
    Returns
    -------
    PSD parameter : np.ndarray
        Pileup discrimination parameter
    peak_values : np.ndarray
        Interpolated peak values.
    fitted amplitudes : np.ndarray
        Estimated amplitude for each pulse.
    peak_positions : np.ndarray
        Sub-sample positions of the peak.
    """
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    transferfunc_max = transferfunc if transferfunc_max is None else transferfunc_max
    mean_dif = np.diff(np.append(meanpulse, meanpulse[-1]))
    fmeanpulse = get_filtered_pulse(mean_dif, transferfunc)
    freq = np.fft.fftfreq(4000, 1e-4)
    peak_values = []
    peak_positions = []
    coeffs = []
    for pulses in loader:
        if hasattr(pulses, 'numpy'):
            pulses = pulses.numpy()
        n_p, win_size = np.shape(pulses)
        row_indices = np.arange(n_p)[:, None]
        # Compute first derivative (difference)
        dpulses = np.diff(np.append(pulses, pulses[:, -1][:, None], axis = 1), axis = 1)
        f_pulse = np.fft.fft(dpulses, axis = 1)
        fpulses = np.real(np.fft.ifft(f_pulse * transferfunc_max))

        # Window of points around the peak to interpolate
        x = np.arange(-5, 6) + arg_max
        y = fpulses[:, arg_max - 5: arg_max + 6]

        # Interpolation function for sub-sample peak finding
        interp_func = interp1d(x, y, kind='cubic', axis=1, bounds_error=False, fill_value='extrapolate')
        x_inter = np.arange(-jitter_max, jitter_max+subsample_size, subsample_size) + arg_max
        inter_fpulse = interp_func(x_inter)
        arg_max_inter_sub = np.argmax(inter_fpulse, axis = 1)

        dt = (x_inter[arg_max_inter_sub] - arg_max) * 1e-4
        f_pulse = f_pulse * np.exp(1j * 2 * np.pi * freq * dt[:, None])
        fpulses = np.real(np.fft.ifft(f_pulse * transferfunc))
        coeff = np.polyfit(
            fmeanpulse[arg_max - jitter_fit:arg_max + jitter_fit + 1],
            fpulses[:, arg_max - jitter_fit:arg_max + jitter_fit + 1].T,
            1
        ).T
        # for f in fpulses[:10]:
        #     plt.scatter(fmeanpulse[arg_max - jitter_fit:arg_max + jitter_fit + 1],f[arg_max - jitter_fit:arg_max + jitter_fit + 1])
        #     plt.show()

        peak_values.append(fpulses[:,arg_max])
        # peak_values.append(np.max(inter_fpulse,axis=1))
        peak_positions.append(x_inter[arg_max_inter_sub])
        coeffs.append(coeff[:, 0])
    coeffs = np.concatenate(coeffs)
    peak_values = np.concatenate(peak_values)
    peak_positions = np.concatenate(peak_positions)
    return (
        peak_values/coeffs, # PSD parameter
        peak_values,  # peak values
        coeffs, # amplitude coefficients
        peak_positions  # sub-sample peak positions
    )
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
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    freq = np.fft.fftfreq(4000, 1e-4)
    dt = (arg_max - 1999)*1e-4
    dt_split = [dt[i:i + batch_size] for i in range(0, len(dt), batch_size)]
    chi2s = []
    for pulses, dt_sub in zip(loader,dt_split):
        # Compute FFT of the pulse derivatives
        diff_pulse = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        f_pulse = np.fft.fft(diff_pulse, axis=1) * transferfunc
        f_pulse = f_pulse / np.linalg.norm(f_pulse, axis=1)[:, None]
        # Chi-square distance between phase-shifted pulses and reference
        chi2 = np.sum(
            np.abs(
                f_pulse* np.exp(1j * 2 * np.pi * freq * dt_sub[:, None]) - f_meanpulse[None, :]
            ) ** 2,
            axis=1
        )
        chi2s.append(chi2)
    chi2s = np.concatenate(chi2s)
    return chi2s
def get_interpolate_chi2_stat(dataset, transferfunc, arg_max, meanpulse):
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
    # f_meanpulse = np.fft.fft(np.concatenate([np.zeros(1000), mean_dif, np.zeros(1000)])) * transferfunc
    freq = np.fft.fftfreq(4000, 1e-4)
    batch_size = 256
    dt = (arg_max - 1999) * 1e-4
    dt_split = [dt[i:i + batch_size] for i in range(0, len(dt), batch_size)]
    sel = (freq > 500) & (freq < 1785)
    df = 1e4 / 2000
    dof = np.sum(sel)
    f_meanpulse =  (np.fft.fft(np.pad(mean_dif,(1000,1000),mode="constant",constant_values=(0,0)))*transferfunc)[sel]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chi2s_stat_per_dof = []
    for pulses, dt_sub in zip(loader,dt_split):
        # Compute FFT of the pulse derivatives
        diff_pulse = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        f_pulse = np.fft.fft(diff_pulse, axis=1)* np.exp(-1j * 2 * np.pi * freq * dt_sub[:, None])*transferfunc
        f_pulse=f_pulse[:,sel]
        A = np.sum(np.conj(f_meanpulse) * f_pulse, axis=1) / np.sum(np.abs(f_meanpulse)**2)
        # Rescale model for each pulse
        M_scaled = A[:, np.newaxis] * f_meanpulse[np.newaxis, :]  # shape (batch, 2000)

        # Compute chi-square per pulse
        residuals = f_pulse - M_scaled  # shape (batch, 2000)
        chi2_stat = np.sum(np.abs(residuals) ** 2, axis = 1) * df
        chi2_stat_per_dof = chi2_stat / dof  # shape (batch,)
        chi2s_stat_per_dof.append(chi2_stat_per_dof)
    chi2s_stat_per_dof = np.concatenate(chi2s_stat_per_dof)
    return chi2s_stat_per_dof


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