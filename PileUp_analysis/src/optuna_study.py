import numpy as np
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import PileUp_analysis.utility.functions as fn
import PileUp_analysis.src.analysis as an
import PileUp_analysis.src.dataset as ds

from optuna.storages import RDBStorage
import optuna
def tunning(trial,transf_OF,meanpulse,data_single,data_pileup):
    fs=10000
    freq = np.fft.fftfreq(len(transf_OF),1/fs)
    A = trial.suggest_float("A", 2, 120,log=True)
    f0 = trial.suggest_float("f0", 1, 5000)
    sigma = trial.suggest_float("sigma", 0.1, 5.0)

    transf_PU = transf_OF * fn.modified_gaussian(freq,A,np.log(f0),sigma,1)
    PSD_single, filtered_amp_single, _, _ = an.get_PSD(data_single, transf_PU, meanpulse, transferfunc_max = transf_OF)
    PSD_pileup, filtered_amp_pileup, _, _ = an.get_PSD(data_pileup, transf_PU, meanpulse, transferfunc_max = transf_OF)
    cut = np.percentile(PSD_single, 10)
    rp = np.sum(PSD_pileup < cut) / len(PSD_pileup)
    BI = fn.calculate_BI(rp, 0.8)
    print(A, f0, sigma , BI)
    return BI
def tunning_div(trial,transf_OF,meanpulse,data_single,data_pileup):
    fs=10000
    freq = np.fft.fftfreq(len(transf_OF),1/fs)
    A = trial.suggest_float("A", 2, 120,log=True)
    f0 = trial.suggest_float("f0", 1, 5000)
    sigma = trial.suggest_float("sigma", 0.1, 5.0)

    transf_PU = transf_OF / fn.modified_gaussian(freq,A,np.log(f0),sigma,1)
    PSD_single, filtered_amp_single, _, _ = an.get_PSD(data_single, transf_PU, meanpulse, transferfunc_max = transf_OF)
    PSD_pileup, filtered_amp_pileup, _, _ = an.get_PSD(data_pileup, transf_PU, meanpulse, transferfunc_max = transf_OF)
    cut = np.percentile(PSD_single, 10)
    rp = np.sum(PSD_pileup < cut) / len(PSD_pileup)
    BI = fn.calculate_BI(rp, 0.8)
    print(A, f0, sigma , BI)
    return BI
channel = int(sys.argv[1])
# channel = 3
database_url = "sqlite:///" + ds.main_dir + f"/channel{channel}/study_channel{channel}.db"
storage = RDBStorage(url=database_url)
# search_space = {"A": np.arange(1, 20,0.5).tolist(), "f0": np.arange(100, 500,10).tolist(), "sigma":np.arange(0.4, 1.2,0.2).tolist()}
# search_space = {"A": np.exp(np.linspace(np.log(2), np.log(120), 20)).tolist(), "f0": np.exp(np.linspace(np.log(2000), np.log(5001), 20)).tolist(),
#                     "sigma": np.arange(0.6, 3.0, 0.2).tolist()}
search_space = {"A": np.exp(np.linspace(np.log(1), np.log(120), 20)).tolist(), "f0": np.exp(np.linspace(np.log(1), np.log(500), 20)).tolist(),
                    "sigma": [0.6]}

try:
    optuna.delete_study(study_name=f"study_channel{channel}_divgauss", storage=database_url)
except KeyError:
    pass  # study doesn't exist yet — that's fine
new_study = optuna.create_study(storage=storage, study_name=f"study_channel{channel}_divgauss", direction='minimize',
                                load_if_exists=True, sampler=optuna.samplers.GridSampler(search_space))

meanpulse, bslspec = ds.get_meanpulse_ANPS(channel)

transf_OF,_ = an.compute_transfer_simple(meanpulse,bslspec)
data_single,data_pileup,_,_,_ = ds.create_data_sets(channel,-1,pos_prefix="_bis")
new_study.optimize(lambda x: tunning_div(x,transf_OF, meanpulse,data_single,data_pileup), n_trials=3000)

