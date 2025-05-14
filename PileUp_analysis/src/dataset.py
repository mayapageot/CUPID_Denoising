import numpy as np
from torch.utils.data import Dataset
import optuna
from dotenv import load_dotenv
import os
load_dotenv()
main_dir = os.getenv("DATA_PATH")
# Define a custom PyTorch dataset for loading and processing binary data


def write_data_stream(file_path, data: np.ndarray) -> None:
    data = (((data + 10.24) * (2 ** 23) / 10.24).astype(np.uint32) << 8)
    with open(file_path, "wb") as f:
        data.tofile(f)
    print("Derivative data saved to:", file_path)


def differenciate_data(channel,delta):
    dir_dif = main_dir + f"/channel{channel}/dif/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_000_dif.bin"
    dir_raw = main_dir + f"/channel{channel}/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_000.bin"
    bin_file = np.fromfile(dir_raw, dtype = np.uint32, offset = 4)
    bin_file = ((np.right_shift(bin_file, 8).astype(np.float32) / (2 ** 23)) * 10.24 - 10.24).astype(np.float32)
    bin_file_dif = np.diff(bin_file)
    write_data_stream(dir_dif, bin_file_dif)
def modify_pos(channel, delta):
    partial = 0
    path_pos_single =main_dir + f"/channel{channel}/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}.pos"
    pos_file_single = np.loadtxt(path_pos_single)[:15999]
    positions_single = pos_file_single[:, 0].astype(int)
    path_trig = main_dir + f"/channel{channel}/dif/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin.edtrigger"
    pos_trig = np.loadtxt(path_trig)[:, 0]
    pos_in_trig = np.searchsorted(pos_trig, positions_single)
    dif = pos_trig[pos_in_trig[pos_in_trig - 1 < len(pos_trig)] - 1] - positions_single[pos_in_trig - 1 < len(pos_trig)]
    pos_file_single[np.argwhere(pos_in_trig - 1 < len(pos_trig))[dif > -60], 0] = np.reshape(
        pos_trig[pos_in_trig[pos_in_trig - 1 < len(pos_trig)] - 1][dif > -60], (-1, 1))
    np.savetxt(
         main_dir + f"/channel{channel}/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}_bis.pos",
        pos_file_single)
class BinaryDataset(Dataset):
    def __init__(self, file_path, win_length, positions, win_shift=0, dtype=np.uint32):
        """
        Initializes the dataset.

        Args:
            file_path (str): Path to the binary file.
            win_length (int): Length of each data window to extract.
            positions (list or np.array): List of starting positions in the binary file for windows.
            win_shift (int): Optional shift applied to window start positions (default: 0).
            dtype (numpy dtype): Data type for reading binary data (default: np.uint32).
        """
        self.file_path = file_path
        self.win_length = int(win_length)
        self.positions = positions
        self.win_shift = int(win_shift)
        self.dtype = dtype

        # Memory-map the binary file to allow efficient reading without loading the entire file into memory.
        # Skips the first 4 bytes of the file using offset=4 (often used to skip a header or metadata).
        self.file_map = np.memmap(self.file_path, dtype=dtype, mode='r', offset=4)

    def load_and_process_window(self, file_map, start):
        """
        Loads a window of data from the memory-mapped file and applies processing.

        Args:
            file_map (np.memmap): Memory-mapped binary file.
            start (int): Starting index for the window.

        Returns:
            np.array: Processed data window.
        """
        # Extract a slice of raw data of length `win_length` starting at `start`
        np_data = np.array(file_map[start:start + self.win_length], dtype=self.dtype)

        # Bit-shift each 32-bit value to the right by 8 bits, convert to float64,
        # normalize from raw values to a range between -10.24 and +10.24
        np_data = ((np.right_shift(np_data, 8).astype(np.float64) / (2 ** 23)) * 10.24 - 10.24)

        return np_data

    def __len__(self):
        """
        Returns the total number of data windows in the dataset.
        """
        return len(self.positions)

    def __getitem__(self, index):
        """
        Retrieves and processes a single data window based on the index.

        Args:
            index (int): Index into the list of positions.

        Returns:
            np.array: Processed data window.
        """
        # Calculate the actual start index for the window
        start = self.positions[index] - self.win_shift

        # Load and process the data window
        data = self.load_and_process_window(self.file_map, start)

        return data
def create_data_sets(channel, shift = 0, pos_prefix = "_bis",file_sufixe=""):
    partial = 0 if channel in [17] else 0
    file_path_pileup = main_dir + f"/channel{channel}/dif/pup_n1-d8_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin"+file_sufixe
    path_pos_pileup = main_dir + f"/channel{channel}/pup_n1-d8_000813_20230628T161508_{channel:03}_{partial:03}{pos_prefix}.pos"
    pos_file_pileup = np.loadtxt(path_pos_pileup)[:15000]
    dt_pileup = pos_file_pileup[:,-1]
    dE_pileup = pos_file_pileup[:,1]-pos_file_pileup[:,2]
    positions_pileup = pos_file_pileup[:,0].astype(int)//4
    data_pileup = BinaryDataset(file_path_pileup, 4000, positions_pileup, win_shift= 2000 - shift)
    datas_pileup = [BinaryDataset(file_path_pileup, 4000, positions_pileup[(dt_pileup < i + 1) & (dt_pileup > i)], win_shift= 2000 - shift) for i in range(7)]
    path_pos_single = main_dir + f"/channel{channel}/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}{pos_prefix}.pos"
    pos_file_single = np.loadtxt(path_pos_single)[:15000]
    positions_single = pos_file_single[:,0].astype(int)//4
    file_path_single = main_dir + f"/channel{channel}/dif/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin"+file_sufixe
    data_single = BinaryDataset(file_path_single, 4000, positions_single, win_shift= 2000 - shift)
    return data_single,data_pileup,dt_pileup,dE_pileup,datas_pileup

def get_meanpulse_ANPS(channel,corr_type = "raw"):
    partial = 1 if channel in [17] else 0
    ANPS = np.fromfile(main_dir + f"/channel{channel}/dif/{corr_type}_noise_diff.bin_spec.bin.dif")
    mean = np.fromfile(main_dir + f"/channel{channel}/dif/pup_n1-d0_000813_20230628T161508_{channel:03}_{partial:03}_diff_bis.bin_edmean.bin")
    return mean, ANPS

def get_best_param(channel):
    study = optuna.load_study(study_name=f"study_channel{channel}_divgauss",
                              storage="sqlite:///" + main_dir + f"/study_channel{channel}.db")
    return study.best_params, study.best_value

def from_edtrigger_to_pos(channel,delta):
    partial = 0 if channel == 17 else 0
    path_pos_single = main_dir + f"/channel{channel}/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}.pos"
    pos_file_single = np.loadtxt(path_pos_single)[:15999]
    positions_single = pos_file_single[:, 0].astype(int)
    path_trig = main_dir + f"/channel{channel}/dif/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}_dif.bin.edtrigger"
    pos_trig = np.loadtxt(path_trig)[:, 0]
    pos_in_trig = np.searchsorted(pos_trig, positions_single)
    dif = pos_trig[pos_in_trig[pos_in_trig - 1 < len(pos_trig)] - 1] - positions_single[pos_in_trig - 1 < len(pos_trig)]
    pos_file_single[np.argwhere(pos_in_trig - 1 < len(pos_trig))[dif > -60], 0] = np.reshape(
        pos_trig[pos_in_trig[pos_in_trig - 1 < len(pos_trig)] - 1][dif > -60], (-1, 1))
    np.savetxt(
        main_dir + f"/channel{channel}/pup_n1-d{delta}_000813_20230628T161508_{channel:03}_{partial:03}_bis.pos",
        pos_file_single)
if __name__ == "__main__":
    for delta in [0,8]:
        modify_pos(19, delta)
        # differenciate_data(19, delta)