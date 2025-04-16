import os

import numpy as np

def write_data_stream(file_path, data: np.ndarray) -> None:
    data = (((data + 10.24) * (2 ** 23) / 10.24).astype(np.uint32) << 8)
    with open(file_path, "wb") as f:
        data.tofile(f)
    print("Derivative data saved to:", file_path)
for i in  [9,11,13,15,17,19]:
    for type in ["raw","corrected"]:
        data_label=0
        channel=i
        path_train = f"/feynman/work/projets/cupid/scratch/mpageot/data/trainings/25_02_24_AutoencoderLSTM_9_813_channel{i}/"
        noise_data = np.load(path_train+"test_{}.npy".format(type))
        noise_data = np.diff(noise_data)
        # print(noise_data.shape)
        noise_data = noise_data[:,24:2024]
        noise_data=  noise_data.reshape(-1)
        save_path = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/dif/"
        #os.rename(path_train+"raw_noise_diff.bin",save_path+"raw_noise_diff.bin")
        write_data_stream(save_path+"{}_noise_diff.bin".format(type),noise_data)