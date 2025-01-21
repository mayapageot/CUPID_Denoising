import numpy as np
import torch
import os
import sys
sys.path.append(os.path.relpath("../src"))
import dataloader as dl
from torch.utils.data import DataLoader, random_split
import convAE
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Load the Training infos
path_train = "/feynman/work/projets/cupid/scratch/mpageot/data/trainings/25_01_20_AutoencoderLSTM_9_813_channel19/"
with open(path_train+"InfoDict.json", "r") as out_file:
    InfoDict = json.load(out_file)

# Load dataset used for training
dataset_dict = InfoDict["dataset_dict"]
STFT_dataset = dl.DatasetSTFTFromBinary(**dataset_dict, dtype=np.uint32)

# Split dataset into training, validation, and test sets
data_spliting_dict = dict(seed=42, train_split=0.7, val_split=0.15)
generator1 = torch.Generator().manual_seed(data_spliting_dict["seed"])
train_size = int(data_spliting_dict["val_split"] * len(STFT_dataset))
val_size = int(data_spliting_dict["val_split"] * len(STFT_dataset))
test_size = len(STFT_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(STFT_dataset, [train_size, val_size, test_size],
                                                        generator=generator1)

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
hyperparameters_dict = InfoDict["hyperparameters_dict"]
model = getattr(convAE, InfoDict["model"])(**hyperparameters_dict).to(device)
model.load_state_dict(torch.load(path_train+'model.pth'))
model.eval()

# Infer test data and save the corrected and raw data
# all_corrected = []
# all_raw = []
# with torch.no_grad():
#     for inputs, targets in test_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         corrected = targets-outputs
#         all_corrected.append(corrected.cpu())
#         all_raw.append(targets.cpu())
# all_corrected = torch.cat(all_corrected, dim=0)
# np_corrected = all_corrected.numpy()
# np.save(path_train+"test_corrected.npy", np_corrected)
# all_raw = torch.cat(all_raw, dim=0)
# np_raw = all_raw.numpy()
# np.save(path_train+"test_raw.npy", np_raw)

data_label = 0
channel_out = dataset_dict["channel_out"]
channel = int(dataset_dict["file_paths"][channel_out][-11:-8])
dir_path_inj = "/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/"
dataset_dict["file_paths"][channel_out] = dir_path_inj + "pup_n1-d{}_000813_20230628T161508_{:03}_000.bin".format(data_label,channel)
positions_path = dir_path_inj + "pup_n1-d{}_000813_20230628T161508_{:03}_000.pos.pt".format(data_label, channel)
dataset_dict["win_shift"] = 1024*2
dataset_dict["positions_path"] = positions_path
datasetSTFT_pulses = dl.DatasetSTFTFromBinary(**dataset_dict, dtype=np.uint32)
pulse_loader = DataLoader(datasetSTFT_pulses, batch_size=32, shuffle=False)

# Infer the pulses data
pulses_pred = []
with torch.no_grad():
    for inputs, _ in pulse_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        pulses_pred.append(outputs.cpu())
pulses_pred = torch.cat(pulses_pred, dim=0)
np_pred = pulses_pred.numpy()
np.save(path_train+"pulses_pred_d{}_{:03}.npy".format(data_label,channel), np_pred)

data_label = 8
channel_out = dataset_dict["channel_out"]
channel = int(dataset_dict["file_paths"][channel_out][-11:-8])
dir_path_inj = "/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/"
dataset_dict["file_paths"][channel_out] = dir_path_inj + "pup_n1-d{}_000813_20230628T161508_{:03}_000.bin".format(data_label,channel)
positions_path = dir_path_inj + "pup_n1-d{}_000813_20230628T161508_{:03}_000.pos.pt".format(data_label, channel)
dataset_dict["win_shift"] = 1024*2
dataset_dict["positions_path"] = positions_path
datasetSTFT_pulses = dl.DatasetSTFTFromBinary(**dataset_dict, dtype=np.uint32)
pulse_loader = DataLoader(datasetSTFT_pulses, batch_size=32, shuffle=False)

# Infer the pulses data
pulses_pred = []
with torch.no_grad():
    for inputs, _ in pulse_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        pulses_pred.append(outputs.cpu())
pulses_pred = torch.cat(pulses_pred, dim=0)
np_pred = pulses_pred.numpy()
np.save(path_train+"pulses_pred_d{}_{:03}.npy".format(data_label,channel), np_pred)