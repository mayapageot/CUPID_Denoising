import numpy as np
from scipy.signal.windows import tukey
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.relpath("../utility"))
from functions import polyfit_batch

def compute_transfer_function(fs, points, position, mean, bslspec):
    mean_dif = np.diff(np.append(mean, mean[-1]))

    freq = np.linspace(0, fs - fs / points, points)

    # Window function
    win = tukey(points)

    hd = points // 2

    # Normalize factor 'k'
    fft_mean_win = np.fft.fft(mean_dif * win)
    k = points / np.sum(np.abs(fft_mean_win) ** 2 / bslspec)

    # Transfer function in frequency domain
    shift = np.exp(-1j * 2 * np.pi * freq * (np.ceil(position * points) / fs))
    transf = k * (np.conj(np.fft.fft(mean_dif)) / bslspec) * shift

    # Inverse FFT of transfer function
    itransf = np.real(np.fft.ifft(transf))

    # Concatenated time-domain impulse response with windowing
    win = win
    itransf_conc = np.concatenate([
        itransf[:hd] * win[hd:],  # first half windowed
        np.zeros(points),  # zero padding
        itransf[hd:] * win[:hd]  # second half windowed
    ])

    # Store full transfer function in app
    transferfunc = np.fft.fft(itransf_conc)

    # Build the padded and windowed pulse
    pulse = np.concatenate([
        np.zeros(points // 2),
        mean_dif * win,
        np.zeros(points // 2)
    ])

    PULSE = np.fft.fft(pulse)
    FPULSE = transferfunc * PULSE

    # Output: filtered pulse in time domain
    func_pulse_TF = np.real(np.fft.ifft(FPULSE))
    transferfunc = transferfunc
    return transferfunc, func_pulse_TF
def get_filtered_pulse(dpulses,transferfunc,windowing=tukey):
    t_size = len(transferfunc)
    if dpulses.ndim == 1:
        win_size = len(dpulses)
        win = windowing(win_size)
        if t_size>win_size:
            f_pulse = np.fft.fft(np.concatenate([np.zeros(win_size//2),dpulses,np.zeros(win_size//2)]))
            fpulse = np.real(np.fft.ifft(f_pulse*transferfunc))
        else :
            f_pulse = np.fft.fft(dpulses)
            fpulse = np.real(np.fft.ifft(f_pulse*transferfunc))
    else:
        n_p, win_size = np.shape(dpulses)
        win = windowing(win_size)
        if t_size>win_size:
            dpulses = np.concatenate([np.zeros((n_p,win_size//2)),dpulses,np.zeros((n_p,win_size//2))],axis=1)
            f_pulse = np.fft.fft(dpulses,axis=1)
            fpulse = np.real(np.fft.ifft(f_pulse*transferfunc,axis=1))
        else :
            f_pulse = np.fft.fft(dpulses,axis=1)
            fpulse = np.real(np.fft.ifft(f_pulse*transferfunc,axis=1))
    return fpulse


def get_interpolate_max(dataset, transf, arg_max=1999):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for pulses in loader:
        if hasattr(pulses, 'numpy'):
            pulses = pulses.numpy()

        n_p, win_size = np.shape(pulses)
        row_indices = np.arange(n_p)[:, None]

        # Compute diff
        dpulses = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        fpulses = get_filtered_pulse(dpulses, transf)

        # Fixed x around arg_max
        x = np.arange(-5, 6) + arg_max
        y = fpulses[:, arg_max - 5: arg_max + 6]

        # Create interpolation function (shared x across rows)
        interp_func = interp1d(x, y, kind='cubic', axis=1, bounds_error=False, fill_value='extrapolate')

        # Interpolate around arg_max ±1 with step 0.01
        x_inter = np.arange(-1, 1, 0.01) + arg_max

        inter_fpulse = interp_func(x_inter)
        arg_max_inter = np.argmax(inter_fpulse, axis=1)

        return (
            inter_fpulse[row_indices, arg_max_inter],  # interpolated max values
            x_inter[arg_max_inter]  # corresponding x positions
        )


def get_interpolate_fitamp(dataset, transf, arg_max, meanpulse, jitter=150):
    mean_dif = np.diff(np.append(meanpulse, meanpulse[-1]))
    fmeanpulse = get_filtered_pulse(mean_dif, transf)
    x = np.arange(len(fmeanpulse))
    y = fmeanpulse
    interp_func_mean = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for pulses in loader:
        if hasattr(pulses, 'numpy'):
            pulses = pulses.numpy()
        n_p, win_size = np.shape(pulses)
        row_indices = np.arange(n_p)[:, None]
        # Compute diff
        dpulses = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        fpulses = get_filtered_pulse(dpulses, transf)

        jitter_array = np.arange(-jitter, jitter + 1)
        coeff = polyfit_batch(interp_func_mean(jitter_array[None, :] + arg_max[:, None]),
                              fpulses[:, 1999 - jitter:1999 + jitter + 1], 1)

        return coeff[:, 0]


def get_interpolate_chi2(dataset, transf, arg_max, meanpulse):
    mean_dif = np.diff(np.append(meanpulse, meanpulse[-1]))
    f_meanpulse = np.fft.fft(np.concatenate([np.zeros(1000), mean_dif, np.zeros(1000)]))
    f_meanpulse = f_meanpulse / np.linalg.norm(f_meanpulse)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    freq = np.fft.fftfreq(4000, 1e-4)
    dt = arg_max - 1999
    for pulses in loader:
        diff_pulse = np.diff(np.append(signals, signals[:, -1][:, None], axis=1), axis=1)
        fft = np.fft.fft(diff_pulse, axis=1)
        f_pulse = fft / np.linalg.norm(fft, axis=1)[:, None]
        chi2 = np.sum(
            np.abs(
                f_pulse * np.exp(-1j * 2 * np.pi * freq * dt[:, None]) - f_meanpulse[None, :]
            ) ** 2,
            axis=1
        )
    return chi2

def compute_transfer_simple(mean,ANPS):
    t1,t1_small = compute_transfer_function(10000, 2000, 0.5, mean, ANPS)
    return t1,t1_small