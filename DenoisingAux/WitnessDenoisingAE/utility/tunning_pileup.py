from scipy.special import erf
import numpy as np
from scipy.signal.windows import tukey
import os
import sys
from torch.utils.data import DataLoader, Dataset
from optuna.storages import RDBStorage
import optuna

points = 2000
fs = 10000
freq = np.fft.fftfreq(points, 1 / fs)


def batch_rolling_window(arr, window_size):
    batch_size, seq_length = arr.shape
    shape = (batch_size, seq_length - window_size + 1, window_size)
    strides = (arr.strides[0], arr.strides[1], arr.strides[1])
    return as_strided(arr, shape=shape, strides=strides)


class datatest(Dataset):
    def __init__(self, file_path, win_length, positions, win_shift=0, dtype=np.uint32):
        self.file_path = file_path
        self.win_length = int(win_length)
        self.positions = positions
        self.win_shift = int(win_shift)
        self.dtype = dtype
        self.file_map = np.memmap(self.file_path, dtype=dtype, mode='r', offset=4)

    def load_and_process_window(self, file_map, start):
        np_data = np.array(file_map[start:start + self.win_length], dtype=self.dtype)
        np_data = ((np.right_shift(np_data, 8).astype(np.float64) / (2 ** 23)) * 10.24 - 10.24)
        return np_data

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        start = self.positions[index] - self.win_shift
        data = self.load_and_process_window(self.file_map, start)
        return data


def get_APS_dataset(dataset, transfert_function=None, windowing=tukey):
    transfert_function = np.ones(2000) if transfert_function is None else transfert_function
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    window = windowing(2000)[None, :]
    APS = np.zeros(2000)
    for signals in loader:
        fft = np.fft.fft(np.diff(np.append(signals, signals[:, -1][:, None], axis=1), axis=1)[:, 1000:3000] * window,
                         axis=1) * transfert_function
        PS = np.abs(fft)
        APS = np.median(PS, axis=0)
    return APS


def get_Afft_dataset(dataset, windowing=tukey):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    window = windowing(dataset.win_length // 2)[None, :]
    for signals in loader:
        signals = signals.numpy()[:, 1000:3000]
        # signals = signals/np.max(signals,axis=1)[:,None]
        ds = np.diff(np.append(signals, signals[:, -1][:, None], axis=1), axis=1)
        fft = np.fft.fft((ds) * window, axis=1)
        Afft = np.median(np.abs(fft), axis=0)
        angle = np.median(np.angle(fft), axis=0)
    return Afft * np.exp(1j * angle)


def compute_meanpulse(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for signals in loader:
        print(signals.shape)
        s = signals.numpy()
        meanpulse = np.mean(s / np.max(s, axis=1)[:, None], axis=0)
    return meanpulse


# def compute_transfer_function(fs, points, position, mean, bslspec):
#     mean_dif = np.diff(np.append(mean, mean[-1]))
#     freq = np.fft.fftfreq(points,1/fs)
#     win = tukey(points)
#     k = points / np.sum(np.abs(np.fft.fft(mean_dif * win)) ** 2 / bslspec)
#     transf = k*np.conj(np.fft.fft(mean_dif))/bslspec * np.exp(-1j * 2 * np.pi * freq * (np.ceil(position * points) / fs))
#     # fpulse = np.real(np.fft.ifft(f_pulse*transf))
#     # transf = transf/np.max(fpulse)
#     itransf = np.real(np.fft.ifft(transf))
#     hd = points // 2
#     itransf_conc = np.concatenate([
#     itransf[:hd] * win[hd:],
#     np.zeros(points),
#     itransf[hd:] * win[:hd]
#     ])
#     transferfunc = np.fft.fft(itransf_conc)
#     fpulse = get_filtered_pulse(mean_dif,transferfunc)
#     transferfunc = transferfunc/np.max(fpulse)
#     fpulse_s = get_filtered_pulse(mean_dif,transf)
#     return transferfunc,transf/np.max(fpulse_s)

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


def compute_transfer_function_diff(fs, points, position, mean_fft, mean_PU_fft, bslspec, mean=None):
    win = tukey(points)
    freq = np.fft.fftfreq(points, 1 / fs)
    fft_diff = np.abs(np.abs(mean_fft) - np.abs(mean_PU_fft))
    fft_diff_smooth = gaussian_filter1d(fft_diff, sigma=5)
    k = points / np.sum(np.abs(fft_diff) * np.abs(mean_fft) / bslspec)
    transf = k * np.conj(fft_diff_smooth * np.exp(1j * np.angle(mean_fft))) / bslspec * np.exp(
        -1j * 2 * np.pi * freq * (np.ceil(position * points) / fs))
    itransf = np.real(np.fft.ifft(transf))
    hd = points // 2
    itransf_conc = np.concatenate([
        itransf[:hd] * win[hd:],
        np.zeros(points),
        itransf[hd:] * win[:hd]
    ])
    transferfunc = np.fft.fft(itransf_conc)
    if mean is not None:
        fpulse = get_filtered_pulse(mean, transferfunc)
        transferfunc = transferfunc / np.max(fpulse)
    return transferfunc, transf


def get_fFFT_pulse(pulse, transferfunc):
    t_size = len(transferfunc)
    win_size = len(pulse)
    win = tukey(win_size)
    if t_size > win_size:
        f_pulse = np.fft.fft(np.concatenate([np.zeros(win_size // 2), pulse * win, np.zeros(win_size // 2)]))
        fpulse = np.real(np.fft.ifft(f_pulse * transferfunc))
    else:
        f_pulse = np.fft.fft(pulse * win)
    return f_pulse * transferfunc


def get_filtered_pulse(dpulses, transferfunc, windowing=tukey):
    t_size = len(transferfunc)
    if dpulses.ndim == 1:
        win_size = len(dpulses)
        win = windowing(win_size)
        if t_size > win_size:
            f_pulse = np.fft.fft(np.concatenate([np.zeros(win_size // 2), dpulses, np.zeros(win_size // 2)]))
            fpulse = np.real(np.fft.ifft(f_pulse * transferfunc))
        else:
            f_pulse = np.fft.fft(dpulses)
            fpulse = np.real(np.fft.ifft(f_pulse * transferfunc))
    else:
        n_p, win_size = np.shape(dpulses)
        win = windowing(win_size)
        if t_size > win_size:
            dpulses = np.concatenate([np.zeros((n_p, win_size // 2)), dpulses, np.zeros((n_p, win_size // 2))], axis=1)
            f_pulse = np.fft.fft(dpulses, axis=1)
            fpulse = np.real(np.fft.ifft(f_pulse * transferfunc, axis=1))
        else:
            f_pulse = np.fft.fft(dpulses, axis=1)
            fpulse = np.real(np.fft.ifft(f_pulse * transferfunc, axis=1))
    return fpulse


# def get_filtered_pulse(pulse_dif,transferfunc,fs=1e4, points=4000,position=0.5):
#     PULSE = np.fft.fft(pulse_dif,axis=1)
#     # freq = np.linspace(0, fs - fs /points, points)
#     # shift = np.exp(-1j * 2 * np.pi * freq * (np.ceil(position * points) / fs))
#     # FPULSE = np.abs(transferfunc * PULSE)*shift
#     FPULSE = transferfunc * PULSE
#     func_pulse_TF = np.real(np.fft.ifft(FPULSE,axis=1))
#     return func_pulse_TF

def create_data_sets(channel, shift=0):
    partial = 1 if channel in [17] else 0
    file_path_pileup = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d8_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin"
    path_pos_pileup = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/pup_n1-d8_000813_20230628T161508_{channel:03}_{partial:03}_bis.pos"
    pos_file_pileup = np.loadtxt(path_pos_pileup)[:15000]
    dt_pileup = pos_file_pileup[:, -1]
    dE_pileup = pos_file_pileup[:, 1] - pos_file_pileup[:, 2]
    positions_pileup = pos_file_pileup[:, 0].astype(int) // 4
    # if channel == 3 else 0
    data_pileup = datatest(file_path_pileup, 4000, positions_pileup, win_shift=2000 - shift)
    datas_pileup = [datatest(file_path_pileup, 4000, positions_pileup[(dt_pileup < i + 1) & (dt_pileup > i)],
                             win_shift=2000 - shift) for i in range(7)]
    path_pos_single = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}_bis.pos"
    pos_file_single = np.loadtxt(path_pos_single)[:15000]
    positions_single = pos_file_single[:, 0].astype(int) // 4
    file_path_single = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin"
    data_single = datatest(file_path_single, 4000, positions_single, win_shift=2000 - shift)
    return data_single, data_pileup, dt_pileup, dE_pileup, datas_pileup


def get_meanpulse_ANPS(channel):
    partial = 1 if channel in [17] else 0
    ANPS = np.fromfile(
        f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/raw_noise_diff.bin_spec.bin.dif")
    mean = np.fromfile(
        f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}_diff.bin_edmean.bin")
    return mean, ANPS


def compute_transfer_simple(mean, ANPS):
    t1, t1_small = compute_transfer_function(10000, 2000, 0.5, mean, ANPS)
    return t1, t1_small


def calculate_BI(rej_power: float, dt: float) -> float:
    Activity = 2.93E-03  # Bq (N/s)
    LMO_MASS = 0.28  # kg
    second_in_year = 3600 * 24 * 365
    Probability = 3.41E-04
    correction_factor = 1
    milliseconds_in_second = 1000
    return (correction_factor * Probability * (1 - rej_power) * Activity * Activity /
            LMO_MASS * second_in_year * dt / milliseconds_in_second)


def modified_gaussian(x, A, x0, sigma, b):
    y = np.ones_like(x)
    y[x > 0] = A * np.exp(-(np.log(x[x > 0]) - x0) ** 2 / (2 * sigma ** 2)) + b
    y[x < 0] = A * np.exp(-(np.log(-x[x < 0]) - x0) ** 2 / (2 * sigma ** 2)) + b
    return y


def vect_roll(arr, shifts):
    n_rows, n_cols = arr.shape
    cols = np.arange(n_cols)
    shifted_cols = (cols[None, :] - shifts[:, None]) % n_cols
    rolled = arr[np.arange(n_rows)[:, None], shifted_cols]
    return rolled


def center_max_per_row_vectorized(arr, max_jitter=16):
    num_rows, num_cols = arr.shape
    center_col = num_cols // 2

    # Get max indices and shifts
    max_indices = np.argmax(arr[:, center_col - max_jitter:center_col + max_jitter], axis=1)
    shifts = max_jitter // 2 - max_indices  # (num_rows,)

    return vect_roll(arr, shifts)


def compute_parameter(meanpulse, data, transf, shift=0, max_jitter=16, jitter=150, delta=0):
    loader = DataLoader(data, batch_size=25, shuffle=False)
    win = tukey(2000)
    mean_dif = np.diff(np.append(meanpulse, meanpulse[-1]))
    # mean_dif = mean_dif/np.max(mean_dif)
    fs = 10000
    points = 2000
    position = 0.5
    freq = np.fft.fftfreq(points, 1 / fs)
    # fmeanpulse = get_filtered_pulse(mean_dif,transf)
    mean_dif = np.concatenate([
        np.zeros(points // 2),
        mean_dif * win,
        np.zeros(points // 2)
    ])
    MEAN = np.fft.fft(mean_dif)
    FMEAN = transf * MEAN
    fmeanpulse = np.real(np.fft.ifft(FMEAN))
    fmeanpulse = fmeanpulse[np.argmax(fmeanpulse) - jitter:np.argmax(fmeanpulse) + jitter]
    amps = []
    coeffs = []
    transf_size = len(transf)
    lt = [0] * 2
    for pulses in loader:
        n_p, win_size = np.shape(pulses)
        # pulses_max = 2000 - jitter + np.argmax(pulses[2000-jitter,2000+jitter],axis=1)
        # row_indices = np.arange(pulses.shape[0])[:, np.newaxis]
        # pulses = center_max_per_row_vectorized(pulses.numpy())
        dpulses = np.diff(np.append(pulses, pulses[:, -1][:, None], axis=1), axis=1)
        fpulses = get_filtered_pulse(dpulses,
                                     transf)  # [:,transf_size//2-jitter-max_jitter:transf_size//2+jitter+max_jitter]

        # x = np.arange(transf_size//2-jitter-max_jitter-shift,transf_size//2+jitter+max_jitter+1+shift)
        # interp_func = interp1d(x, fpulses[:,transf_size//2-jitter-max_jitter-shift:transf_size//2+jitter+max_jitter+1+shift], kind='cubic', axis=1)
        # x_inter = np.arange(transf_size//2-max_jitter-jitter-shift,transf_size//2+max_jitter+jitter+shift,0.1)
        # fpulse_inter = interp_func(x_inter)
        # arg_max = np.argmax(fpulse_inter[:,jitter*10:(jitter+2*max_jitter)*10],axis=1)
        # if max_jitter>0:
        #     arg_max = (jitter*10)-(shift*10)+np.argmax(fpulse_inter[:,(jitter-shift)*10:(jitter+2*max_jitter+shift)*10],axis=1)
        # else :
        #      arg_max = ((jitter*10)-(shift*10)) *np.ones(n_p,dtype=int)

        # jitter_array = np.arange(-jitter*10,+jitter*10,10)
        # row_indices = np.arange(fpulse_inter.shape[0])[:, np.newaxis]
        # cf_d = []
        # residuals = []
        # for delay in range(-shift*10,+shift*10+1):
        #     result = fpulse_inter[row_indices, jitter_array+arg_max[:, np.newaxis]+delay]
        #     coeff,res,rank, singular_values, rcond = np.polyfit(fmeanpulse, result.T, 1,full=True)
        #     # for r in result:
        #     #     plt.plot(fmeanpulse*r.max(),lw=0.5)
        #     #     plt.plot(r,lw=0.5)
        #     #     plt.show()
        #     cf_d.append(coeff.T)
        #     residuals.append(res.reshape(1,-1))
        # am = np.argmin(np.concatenate(residuals,axis=0),axis=0)
        # result = fpulse_inter[row_indices, jitter_array+arg_max[:, np.newaxis]]
        if max_jitter > 0:
            arg_max = transf_size // 2 + np.argmax(
                fpulses[:, transf_size // 2 - max_jitter:transf_size // 2 + max_jitter], axis=1)
        else:
            arg_max = (transf_size//2) * np.ones(n_p, dtype=int)
        jitter_array = np.arange(-jitter, jitter) + 1 + delta
        row_indices = np.arange(fpulses.shape[0])[:, np.newaxis]
        result = fpulses[row_indices, jitter_array + arg_max[:, np.newaxis]]
        coeff = np.polyfit(fmeanpulse, result.T, 1).T
        amps.append(fpulses[row_indices, arg_max[:, np.newaxis]].reshape(-1))
        coeffs.append(coeff)
        # amps.append(fpulse_inter[row_indices,arg_max[:, np.newaxis]].reshape(-1))
        # cf_d = np.array(cf_d)
        # coeffs.append(cf_d[am, np.arange(cf_d.shape[1])])
        # break
    amps = np.concatenate(amps)
    coeffs = np.concatenate(coeffs)
    return amps, coeffs[:, 0], coeffs[:, 1]


def compute_rp_from_transfert(transfertfunc, mean, data_single, data_pileup):
    filtered_single, fitted_single, _ = compute_parameter(mean, data_single, transfertfunc)
    PSD_single = filtered_single / fitted_single
    filtered_pileup, fitted_pileup, _ = compute_parameter(mean, data_pileup, transfertfunc)
    PSD_pileup = filtered_pileup / fitted_pileup
    bins = np.linspace(0.9, 1.3, 500)
    plt.hist(PSD_single, bins, alpha=0.8)
    plt.hist(PSD_pileup, bins, alpha=0.8)
    plt.show()
    cut = np.percentile(PSD_single, 10)
    rp = np.sum(PSD_pileup < cut) / len(PSD_pileup)
    return rp


def compute_rp_from_ANPS(bslspec, mean, data_single, data_pileup):
    transfertfunc, _ = compute_transfer_function(10000, 2000, 0.5, mean, bslspec)
    return compute_rp_from_transfert(transfertfunc, mean, data_single, data_pileup)


def compute_rp(channel):
    data_single, data_pileup, dt_pileup, dE_pileup, datas_pileup = create_data_sets(channel)
    mean, ANPS = get_meanpulse_ANPS(channel)
    return compute_rp_from_ANPS(ANPS, mean, data_single, data_pileup)


def get_best_param(channel):
    partial = 1 if channel in [17] else 0
    study = optuna.load_study(study_name=f"study_channel{channel}",
                              storage="sqlite:///" + f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/study_channel{channel}_bruteforce_bis.db")
    return study.best_params, study.best_value, calculate_BI(study.best_value, 0.8)


def plot_MeanPulse_powerspectrum(channel):
    meanpulse, ANPS = get_meanpulse_ANPS(channel)
    transf_raw, _ = compute_transfer_simple(meanpulse, ANPS)
    best_params, _, _ = get_best_param(channel)
    mg = modified_gaussian(freq, best_params["A"], np.log(best_params["f0"]), best_params["sigma"], 1)
    transf_best, _ = compute_transfer_simple(meanpulse, ANPS * mg)
    f_meanpulse = np.fft.fft(meanpulse)
    return np.abs(f_meanpulse)[:1000], np.abs(get_fFFT_pulse(meanpulse, transf_raw))[:2000], np.abs(
        get_fFFT_pulse(meanpulse, transf_best))[:2000], mg, freq


def get_OF_PU_transf(channel):
    meanpulse, ANPS = get_meanpulse_ANPS(channel)
    transf_raw, _ = compute_transfer_simple(meanpulse, ANPS)
    best_params, _, _ = get_best_param(channel)
    mg = modified_gaussian(freq, best_params["A"], np.log(best_params["f0"]), best_params["sigma"], 1)
    transf_best, _ = compute_transfer_simple(meanpulse, ANPS * mg)
    return transf_raw, transf_best


def get_OF_PU_ANPS(channel):
    meanpulse, ANPS = get_meanpulse_ANPS(channel)
    best_params, _, _ = get_best_param(channel)
    mg = modified_gaussian(freq, best_params["A"], np.log(best_params["f0"]), best_params["sigma"], 1)
    return ANPS, ANPS * mg


def get_rp(path_pileup, path_single):
    bins = np.linspace(0.9, 1.1, 200)
    pileup = np.loadtxt(path_pileup, usecols=range(20))[:15001]
    single = np.loadtxt(path_single, usecols=range(20))[:15001]
    PSD_pileup = pileup[:-1, 2] / pileup[:-1, 9]
    PSD_single = single[:-1, 2] / single[:-1, 9]
    plt.hist(PSD_single, bins, alpha=0.8)
    plt.hist(PSD_pileup, bins, alpha=0.8)
    cut = np.percentile(PSD_single, 10)
    rp = np.sum(PSD_pileup < cut) / len(PSD_pileup)
    return rp


def compute_arg_rp(channel, name=""):
    partial = 1 if channel in [17] else 0
    path_pileup = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d8_000813_20230628T161508_{channel:03}_{partial:03}_dif{name}.bin.ntp"
    path_single = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}_dif{name}.bin.ntp"
    return get_rp(path_pileup, path_single)
def tunning(trial,bslspec_base,mean,data_single,data_pileup):
    points=2000
    fs=10000
    freq = np.fft.fftfreq(points, 1 / fs)
    A = trial.suggest_float("A", 0, 1e4)
    f0 = trial.suggest_float("f0", 0, 2e3)
    sigma = trial.suggest_float("sigma", 0, 1e2)
    print(A,f0,sigma)
    bslspec_modified = bslspec_base*modified_gaussian(freq,A,np.log(f0),sigma,1)
    transfertfunc, f_mp = compute_transfer_function(10000, 2000, 0.5, mean, bslspec_modified)
    filtered_single, fitted_single, _ = compute_parameter(mean, data_single, transfertfunc, 0, 0, 150, 0)
    PSD_single = filtered_single / fitted_single
    filtered_pileup, fitted_pileup, _ = compute_parameter(mean, data_pileup, transfertfunc, 0, 0, 150, 0)
    PSD_pileup = filtered_pileup / fitted_pileup
    cut = np.percentile(PSD_single, 10)
    rp = np.sum(PSD_pileup < cut) / len(PSD_pileup)
    return rp
def tunning_test(params,t_base,mean,data_single,data_pileup):
    points=2000
    fs=10000
    freq = np.fft.fftfreq(len(t_base),1/fs)
    A = params[0]
    f0 = params[1]
    sigma = params[2]
    print(A,f0,sigma)
    transfertfunc = t_base * modified_gaussian(freq, A, np.log(f0), sigma, 1)
    filtered_single, fitted_single, _ = compute_parameter(mean, data_single, transfertfunc, 0, 0, 150, 0)
    PSD_single = filtered_single / fitted_single
    filtered_pileup, fitted_pileup, _ = compute_parameter(mean, data_pileup, transfertfunc, 0, 0, 150, 0)
    PSD_pileup = filtered_pileup / fitted_pileup
    cut = np.percentile(PSD_single, 10)
    rp = np.sum(PSD_pileup < cut) / len(PSD_pileup)
    return rp
def tunning_bis(trial,t_base,mean,data_single,data_pileup):
    fs=10000
    freq = np.fft.fftfreq(len(t_base),1/fs)
    A = trial.suggest_float("A", 1, 1e5,log=True)
    f0 = trial.suggest_float("f0", 0, 2e3)
    sigma = trial.suggest_float("sigma", 0, 1e2)
    print(A,f0,sigma)
    transfertfunc = t_base * modified_gaussian(freq,A,np.log(f0),sigma,1)
    filtered_single, fitted_single, _ = compute_parameter(mean, data_single, transfertfunc, 0, 0, 150, 0)
    PSD_single = filtered_single / fitted_single
    filtered_pileup, fitted_pileup, _ = compute_parameter(mean, data_pileup, transfertfunc, 0, 0, 150, 0)
    PSD_pileup = filtered_pileup / fitted_pileup
    cut = np.percentile(PSD_single, 10)
    rp = np.sum(PSD_pileup < cut) / len(PSD_pileup)
    return rp

def copy_matching_trials(old_study, new_study):
    for trial in old_study.trials:
        new_study.add_trial(trial)

if __name__=="__main__":
    # channel = int(sys.argv[1])
    #sbatch --output=pl$n.log --job-name=pl$n.log tunning_pl.sh "$n"
    channel = 11
    print(f"start channel{channel}")
    # for channel in [9,11,13,15,17,19]:
    data_single, data_pileup, dt_pileup, dE_pileup, datas_pileup = create_data_sets(channel, -2)
    mean, bslspec = get_meanpulse_ANPS(channel)
    # database_url_old = "sqlite:///" + f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/study_channel{channel}_bruteforce.db"
    # storage_old = RDBStorage(url=database_url_old)
    # old_study = optuna.load_study(study_name=f"study_channel{channel}",storage=storage_old)
    # database_url_new = "sqlite:///" + f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/study_channel{channel}_gauss_mult_bis.db"
    # storage_new = RDBStorage(url=database_url_new)
    # search_space = {"A": np.arange(1, 20,0.5).tolist(), "f0": np.arange(100, 500,10).tolist(), "sigma":np.arange(0.4, 1.2,0.2).tolist()}
    # # search_bis = {"A": np.exp(np.arange(1, 5, 0.2)).tolist(), "f0": np.arange(60, 500, 20).tolist(),
    # #                 "sigma": np.arange(0.3, 1.2, 0.1).tolist()}
    # new_study = optuna.create_study(storage=storage_new, study_name=f"study_channel{channel}", direction='maximize',
    #                             load_if_exists=True, sampler=optuna.samplers.GridSampler(search_bis))
    # copy_matching_trials(old_study, new_study)
    # t_base, _ =  compute_transfer_function(10000, 2000, 0.5, np.diff(np.append(mean, mean[-1])), bslspec_base)
    t_base, _ = compute_transfer_function(10000, 2000, 0.5, mean, bslspec)
    # new_study.optimize(lambda x: tunning_bis(x,t_base, mean,data_single,data_pileup), n_trials=1000)
    print(tunning_test([50,700,3.],t_base, mean,data_single,data_pileup))
    # new_study.optimize(lambda x: tunning(x, bslspec, mean, data_single, data_pileup), n_trials=5000)
    # print(tunning_test([5,140,0.4], bslspec, mean, data_single, data_pileup))

