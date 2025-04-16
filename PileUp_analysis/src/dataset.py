import numpy as np
from torch.utils.data import Dataset
import optuna
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

def create_data_sets(channel,shift = 0,pos_prf = "_bis"):
    partial = 1 if channel in [17] else 0
    file_path_pileup = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d8_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin"
    path_pos_pileup = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/pup_n1-d8_000813_20230628T161508_{channel:03}_{partial:03}{pos_prf}.pos"
    pos_file_pileup = np.loadtxt(path_pos_pileup)[:15000]
    dt_pileup = pos_file_pileup[:,-1]
    dE_pileup = pos_file_pileup[:,1]-pos_file_pileup[:,2]
    positions_pileup = pos_file_pileup[:,0].astype(int)//4
    data_pileup = datatest(file_path_pileup, 4000, positions_pileup,win_shift=2000-shift)
    datas_pileup = [datatest(file_path_pileup, 4000, positions_pileup[(dt_pileup<i+1)&(dt_pileup>i)],win_shift=2000-shift) for i in range(7)]
    path_pos_single = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}{pos_prf}.pos"
    pos_file_single = np.loadtxt(path_pos_single)[:15000]
    positions_single = pos_file_single[:,0].astype(int)//4
    file_path_single = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin"
    data_single = datatest(file_path_single, 4000, positions_single ,win_shift=2000-shift)
    return data_single,data_pileup,dt_pileup,dE_pileup,datas_pileup

def get_meanpulse_ANPS(channel):
    partial = 1 if channel in [17] else 0
    ANPS = np.fromfile(f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/raw_noise_diff.bin_spec.bin.dif")
    mean = np.fromfile(f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}_diff_bis.bin_edmean.bin")
    return mean, ANPS

def get_best_param(channel):
    partial = 1 if channel in [17] else 0
    study = optuna.load_study(study_name=f"study_channel{channel}",
                              storage="sqlite:///" + f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/study_channel{channel}_gauss_spec_bis.db")
    return study.best_params, study.best_value

def from_edtrigger_to_pos(channel,delta):
    partial = 1 if channel == 17 else 0
    path_pos_single = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}.pos"
    pos_file_single = np.loadtxt(path_pos_single)[:15999]
    positions_single = pos_file_single[:, 0].astype(int)
    path_trig = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin.edtrigger"
    pos_trig = np.loadtxt(path_trig)[:, 0]
    pos_in_trig = np.searchsorted(pos_trig, positions_single)
    dif = pos_trig[pos_in_trig[pos_in_trig - 1 < len(pos_trig)] - 1] - positions_single[pos_in_trig - 1 < len(pos_trig)]
    pos_file_single[np.argwhere(pos_in_trig - 1 < len(pos_trig))[dif > -60], 0] = np.reshape(
        pos_trig[pos_in_trig[pos_in_trig - 1 < len(pos_trig)] - 1][dif > -60], (-1, 1))
    np.savetxt(
        f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}_bis.pos",
        pos_file_single)