import numpy as np
import torch
import json
def write_data_stream(file_path, data: np.ndarray) -> None:
    data = (((data + 10.24) * (2 ** 23) / 10.24).astype(np.uint32) << 8)
    with open(file_path, "wb") as f:
        data.tofile(f)
    print("Derivative data saved to:", file_path)
for i in [11,13,15,17]:
    partial_num = 1 if i == 17 else 0
    path_train = f"/feynman/work/projets/cupid/scratch/mpageot/data/trainings/25_02_24_AutoencoderLSTM_9_813_channel{i}/"
    with open(path_train + "InfoDict.json", "r") as out_file:
        InfoDict = json.load(out_file)

    partial_num = 1 if i == 17 else 0
    # Load dataset used for training
    dataset_dict = InfoDict["dataset_dict"]
    for data_label in [0, 8]:
        channel_out = dataset_dict["channel_out"]
        channel = int(dataset_dict["file_paths"][channel_out][-11:-8])
        dir_path_inj = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/"
        if partial_num != 0:
            dataset_dict["file_paths"] = [
                f"/feynman/work/projets/cupid/data/LSM/RUN9/000813_20230628T161508_{ch:03}_{partial_num:03}.bin" for ch
                in [9, 11, 13, 15, 17, 19]]
        file_path =  dir_path_inj + "pup_n1-d{}_000813_20230628T161508_{:03}_{:03}.bin".format(data_label,
                                                                                                     channel,
                                                                                                     partial_num)
        dataset_dict["file_paths"][channel_out] =file_path
        positions_path = dir_path_inj + "pup_n1-d{}_000813_20230628T161508_{:03}_{:03}.pos_selected.pt".format(
            data_label, channel, partial_num)
        path_pred = path_train+"pulses_pred_d{}_{:03}.npy".format(data_label,channel)
        win_shift = 1024*2
        globalStd = InfoDict["dataset_dict"]["globalStd"][InfoDict["dataset_dict"]["channel_out"]]
        start_window = dataset_dict["start_window"]
        bin_file = np.fromfile(file_path, dtype=np.uint32, offset=4)
        bin_file = ((np.right_shift(bin_file, 8).astype(np.float32) / (2 ** 23)) * 10.24 - 10.24).astype(np.float32)
        pred_file = np.load(path_pred)*globalStd
        positions = torch.load(positions_path).numpy()
        n_sample,l = pred_file.shape
        n_pos = positions.shape
        indices = np.arange(l)[:, None] + positions - win_shift + start_window
        bin_file[indices.T] = bin_file[indices.T] - pred_file

        bin_file.astype(np.uint32).tofile(file_path+"_corrected.bin")
        write_data_stream(file_path+"_corrected.bin", bin_file)
