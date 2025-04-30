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
from optuna.storages import RDBStorage
import optuna

def progress_bar(iteration, total, loss, loss_val, length=40):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r[{bar}] {percent:.1f}% train_loss: {loss:.2f} val_loss: {loss_val:.2f}')
    sys.stdout.flush()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def create_data_sets(dataset_dict, data_spliting_dict):
    STFT_dataset = dl.DatasetSTFTFromBinary(**dataset_dict, dtype=np.uint32)
    dataset_dict["positions_path"] = STFT_dataset.positions_path
    dataset_dict["globalStd"] = STFT_dataset.globalStd.tolist()
    if "batch_size" not in data_spliting_dict:
        data_spliting_dict["batch_size"] = 32
    # Split dataset into training, validation, and test sets
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
    return dataset_dict, train_loader, val_loader, test_loader

def train(dir_train, num_epochs=50, InfoDict=None, valiation_step=-1):
    os.makedirs(dir_train, exist_ok=True)
    if InfoDict is None:
        if not os.path.isfile(os.path.join(dir_train, "InfoDict.json")):
            raise FileNotFoundError(
                f"no InfoDict in {dir_train}. Give a InfoDict for a first training"
            )
        with open(os.path.join(dir_train, "InfoDict.json"), "r") as out_file:
            InfoDict = json.load(out_file)
    dataset_dict = InfoDict["dataset_dict"]
    data_spliting_dict = InfoDict["data_spliting_dict"]
    hyperparameters_dict = InfoDict["hyperparameters_dict"]
    train_dict = InfoDict["train_dict"]
    # Load the model
    model = getattr(convAE, InfoDict["model"])(**hyperparameters_dict).to(device)
    if model.latent_space_size < 2:
        raise optuna.TrialPruned()
    dataset_dict, train_loader, val_loader, test_loader = create_data_sets(dataset_dict, data_spliting_dict)
    criterion = getattr(nn, train_dict["loss_function"] + "Loss")()
    optimizer = getattr(optim, train_dict["optimizer"])(model.parameters(), lr=train_dict["lr"])
    # Training loop
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if i % valiation_step == 1 or i==len(train_loader)-1:
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
        train_dict["num_epochs"]+=1
        InfoDict = dict(model=model.__class__.__name__,
                        dataset_dict=dataset_dict,
                        data_spliting_dict=data_spliting_dict,
                        hyperparameters_dict=hyperparameters_dict,
                        train_dict=train_dict)
        with open(os.path.join(dir_train, "InfoDict.json"), "w") as out_file:
            json.dump(InfoDict, out_file)
        print("InfoDict saved.")
    return train_dict["training_loss"], train_dict["validation_loss"]

def better_suggest(trial, name, exemple_data, low ,high, *args):
    data_type = type(exemple_data)
    if data_type is int:
        return trial.suggest_int(name, low, high, *args)
    elif data_type is float:
        return trial.suggest_float(name, low, high,*args)
    elif data_type is list:
        for i in range(len(exemple_data)):
            exemple_data[i] = better_suggest(trial, name+"_{}".format(i), exemple_data[i], low[i] ,high[i], *args)
        return exemple_data

def tune_hyper(dir_trainings, n_trials=10, InfoDictbase=None, opt_parameters_dict=None, num_epochs=10, create_study=False):
    dir_trainings = os.path.dirname(dir_trainings)
    os.makedirs(dir_trainings, exist_ok=True)
    study_name = dir_trainings.split("/")[-1]
    database_url = "sqlite:///"+os.path.join(dir_trainings, "study.db")
    # Create an Optuna study with RDBStorage
    storage = RDBStorage(url=database_url)
    if os.path.isfile(os.path.join(dir_trainings, "InfoDictbase.json")):
        # Load already created study
        with open(os.path.join(dir_trainings,"InfoDictbase.json"), "r") as out_file:
            InfoDictbase = json.load(out_file)
        if opt_parameters_dict is None:
            opt_parameters_dict = InfoDictbase["opt_parameters_dict"]
        num_epochs = InfoDictbase["num_epochs_study"]
    elif InfoDictbase is None or opt_parameters_dict is None:
        raise FileNotFoundError(
            "Create a study first by passing InfoDictbase and opt_parameters_dict to tune_hyper"
        )
    else:
        # Save study parameters
        InfoDictbase["opt_parameters_dict"] = opt_parameters_dict
        InfoDictbase["num_epochs_study"] = num_epochs
        with open(os.path.join(dir_trainings,"InfoDictbase.json"), "w") as out_file:
            json.dump(InfoDictbase, out_file)

    def objective(trial):
        # Define the parameters to opimize
        InfoDict = InfoDictbase.copy()
        print(opt_parameters_dict)
        for opt_parameter in opt_parameters_dict:
            if opt_parameter == "n_conv_layer":
                n_conv_layer = trial.suggest_int(opt_parameter, *opt_parameters_dict[opt_parameter])
                InfoDict["hyperparameters_dict"]['filters']=[2**(i+4) for i in range(n_conv_layer)]
            else:
                InfoDict["hyperparameters_dict"][opt_parameter] = better_suggest(trial,opt_parameter,
                                                                                 InfoDict["hyperparameters_dict"][opt_parameter],
                                                                                 *opt_parameters_dict[opt_parameter])
                if opt_parameter in InfoDict["dataset_dict"]:
                    InfoDict["dataset_dict"][opt_parameter] = InfoDict["hyperparameters_dict"][opt_parameter]
        print(InfoDict["hyperparameters_dict"])
        dir_train = os.path.join(dir_trainings,"trial_{}".format(trial.number))
        InfoDict["train_dict"]["path_state_dict"] = os.path.join(dir_train, 'model.pth')
        os.makedirs(dir_train, exist_ok=True)
        with open(os.path.join(dir_train, "InfoDict.json"), "w") as out_file:
            json.dump(InfoDict, out_file)
        model = getattr(convAE, InfoDict["model"])(**InfoDict["hyperparameters_dict"])
        if model.latent_space_size < 2:
            raise optuna.TrialPruned()
        del model
        _,validation_loss = train(dir_train, num_epochs=num_epochs)
        return validation_loss
    study = optuna.create_study(storage=storage, study_name=study_name, direction='minimize', load_if_exists=True)
    if create_study:
        print("study created at ", dir_trainings)
    else :
        study.optimize(objective, n_trials=n_trials)

if __name__=="__main__":
    # Create the dataset
    dir_path = "/feynman/work/projets/cupid/data/LSM/RUN9/"
    RUN = 9
    meas = 813
    name_meas = "20230628T161508"
    channel_list = [3,5,9,11,13,15,17,19]
    file_paths = [dir_path + "000813_20230628T161508_{:03}_000.bin".format(i) for i in channel_list]
    n_channel = 2 * len(file_paths)
    for channel_guess in range(len(channel_list)):
        channel = int(file_paths[channel_guess][-11:-8])
        win_length = 4096
        n_fft = 320
        out_window_size = 2048
        hop_length = 64
        positions_path = ("/feynman/work/projets/cupid/data/LSM/RUN9/positions/pos_start_"+
                          "_".join([str(I) for I in channel_list])+".pt")
        positions_path = positions_path if os.path.isfile(positions_path) else None
        #globalStd= [46491.36328125, 58046.87109375, 96895.8984375, 75740.21875, 101882.8359375, 84200.265625, 61979.71875, 236090.34375]
        #globalStd = None
        globalStd_path = ("/feynman/work/projets/cupid/data/LSM/RUN9/positions/globalStd_"+
                          "_".join([str(I) for I in channel_list])+".pt")

        globalStd = None if not os.path.isfile(globalStd_path) else torch.load(globalStd_path).tolist()
        dataset_dict = dict(file_paths=file_paths, positions_path=positions_path, win_length=win_length,
                            start_window=1024, end_window=1024 * 3, channel_out=channel_guess,
                            n_fft=n_fft, hop_length=hop_length, globalStd=globalStd)
        data_spliting_dict = dict(seed=42, train_split=0.7, val_split=0.15)
        model_name = "AutoencoderLSTM"
        hyperparameters_dict = dict(filters=[16, 32, 64, 128, 256], kernel_size=(1, 3), strides=(1, 2),
                                    n_LSTM=2, LSTM_group_number=4,
                                    hop_length=hop_length, win_length=win_length, n_fft=n_fft,
                                    out_window_size=out_window_size,
                                    n_in=n_channel, n_out=1, unknown_window_center=0.5)

        dir_train = ("/feynman/work/projets/cupid/scratch/mpageot/data/trainings/{}_{}_{}_{}_channel{}/".
                     format(datetime.date.today().strftime("%y_%m_%d"), model_name, RUN, meas, channel))
        train_dict = dict(path_state_dict=dir_train + 'model.pth', num_epochs=0, optimizer="Adam", lr=0.001,
                          loss_function="MSE", device=str(device))
        InfoDict = dict(model=model_name,
                        dataset_dict=dataset_dict,
                        data_spliting_dict=data_spliting_dict,
                        hyperparameters_dict=hyperparameters_dict,
                        train_dict=train_dict)
        os.makedirs(dir_train, exist_ok=True)
        with open(os.path.join(dir_train, "InfoDict.json"), "w") as out_file:
            json.dump(InfoDict, out_file)
        print(dir_train)
        train(dir_train, num_epochs=10)
    # opt_parameters_dict = dict(LSTM_group_number=[16,128],n_fft=[384,1024,32], n_conv_layer=[4,7])
    # dir_trainings = ("/feynman/work/projets/cupid/scratch/mpageot/data/trainings/opt_{}_RUN{}_{}_channel{}_4/".
    #                  format(model_name,RUN, meas, channel))
    # tune_hyper(dir_trainings,1,InfoDictbase=InfoDict,opt_parameters_dict=opt_parameters_dict, create_study=True)
