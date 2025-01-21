import json
import numpy as np
import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
import datetime
sys.path.append(os.path.relpath("../src"))
import dataloader as dl
from torch.utils.data import DataLoader, random_split
import convAE


def progress_bar(iteration, total, loss, loss_val, length=40):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r[{bar}] {percent:.1f}% train_loss: {loss:.2f} val_loss: {loss_val:.2f}')
    sys.stdout.flush()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Create the dataset
dir_path = "/feynman/work/projets/cupid/data/LSM/RUN9/"
RUN = 9
meas = 813
name_meas = "20230628T161508"
file_paths = [dir_path + "{:06}_{}_{:03}_000.bin".format(meas,name_meas,i) for i in [3, 5, 9, 11, 13, 15, 17, 19]]
n_channel = 2 * len(file_paths)
channel_guess = 7
channel = int(file_paths[channel_guess][-11:-8])
win_length = 4095
n_fft = 320
start_window = 1024
end_window = 1024 * 3
out_window_size = end_window - start_window
hop_length = 64

positions_path = "/feynman/work/projets/cupid/scratch/mpageot/data/pos_start_813_3_5_9_11_13_15_17_19.pt"
dataset_dict = dict(file_paths=file_paths, positions_path=positions_path, win_length=win_length,
                    start_window=1024, end_window=1024 * 3, channel_out=channel_guess,
                    n_fft=n_fft, hop_length=hop_length)
STFT_dataset = dl.DatasetSTFTFromBinary(**dataset_dict, dtype=np.uint32)
dataset_dict["positions_path"] = STFT_dataset.positions_path
dataset_dict["globalStd"] = STFT_dataset.globalStd.tolist()
print(dataset_dict["globalStd"] )
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

# Load the model
hyperparameters_dict = dict(filters=[16, 32, 64, 128, 256], kernel_size=(1, 3), strides=(1, 2),
                            n_LSTM=2, LSTM_group_number=4,
                            hop_length=hop_length, win_length=win_length, n_fft=n_fft, out_window_size=out_window_size,
                            n_in=n_channel, n_out=1)
model = convAE.AutoencoderLSTM(**hyperparameters_dict).to(device)
# hyperparameters_dict = dict(filters=[16, 32, 64, 128, 256], kernel_size=(1, 3), strides=(1, 2),
#                             hop_length=hop_length, win_length=win_length, n_fft=n_fft, out_window_size=out_window_size,
#                             n_in=n_channel, n_out=1)
# model = convAE.AutoencoderSkipLayers(**hyperparameters_dict).to(device)

# Training loop
path_train = ("/feynman/work/projets/cupid/scratch/mpageot/data/trainings/{}_{}_{}_{}_channel{}/".
              format(datetime.date.today().strftime("%y_%m_%d"),model.__class__.__name__,RUN,meas,channel))
if not os.path.isdir(path_train):
    os.mkdir(os.path.dirname(path_train))
train_dict = dict(path_state_dict=path_train + 'model.pth', num_epochs=20, optimizer="Adam", lr=0.001,
                  loss_function="MSE", device=str(device))
num_epochs = train_dict["num_epochs"]
criterion = getattr(nn, train_dict["loss_function"] + "Loss")()
optimizer = getattr(optim, train_dict["optimizer"])(model.parameters(), lr=train_dict["lr"])

InfoDict = dict(model=model.__class__.__name__,
                dataset_dict=dataset_dict,
                data_spliting_dict=data_spliting_dict,
                hyperparameters_dict=hyperparameters_dict,
                train_dict=train_dict)
with open(path_train+"InfoDict.json", "w") as out_file:
    json.dump(InfoDict, out_file)
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i % 4 == 0:
            with torch.no_grad():
                loss_val = 0
                for inputs_val, targets_val in val_loader:
                    inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                    outputs_val = model(inputs_val)
                    loss_val += criterion(outputs_val, targets_val).item()
                loss_val = loss_val/len(val_loader)
            progress_bar(i + 1, len(train_loader), loss.item(),loss_val)
    print(f"\nEpoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    # Save model parameters
    torch.save(model.state_dict(), train_dict["path_state_dict"])
    print("Model parameters saved.")
train_dict["training_loss"] = loss.item()
train_dict["validation_loss"] = loss_val
with open(path_train+"InfoDict.json", "w") as out_file:
    json.dump(InfoDict, out_file)