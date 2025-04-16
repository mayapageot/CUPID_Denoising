import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import IterableDataset
from contextlib import contextmanager
import os

class DatasetFromNPZ(Dataset):
    def __init__(self, 
                 npzFile,
                 batchSize: int,
                 preprocess=False,
                 savePath=None,
                 standardizedDataPath=None):
        self.standardizedDataPath = standardizedDataPath
        if(standardizedDataPath is not None):
            self.loadStandardizedData()
            preprocess = False
        else:
            data = np.load(npzFile)
            self.data = data['noise_event_array']
            self.preprocessed = False
            self.savePath = savePath
            self.batchSize = batchSize
            # self.X = self.data[:,:,1:] # witness channel output
            # self.y = self.data[:,:,0] # bolometer output

            # Assert that each example is of the same length
            # assert X.shape[-1] == y.shape[-1]
            
            # Mean and Std for datasets
            self.globalMean = None
            self.globalStd = None
        
        print(f"Shape of self.data: {self.data.shape}") 

        # Preprocess datasets
        if preprocess:
            print('Preprocessing')
            self.computeGlobalStats(batchSize=self.batchSize)
            self.standardizeData(save=True)
            # self.X = self.data[:,:,1:] # witness channel output
            # self.y = self.data[:,:,0] # bolometer output
            print(f"Shape of self.data: {self.data.shape}")
            # print(f"self.X shape: {self.X.shape}")
            # print(f"self.y shape: {self.y.shape}")

        # else:
        #     self.X = self.data[:,:,1:] # witness channel output
        #     self.y = self.data[:,:,0] # bolometer output

        # Assert that each example is of the same length
        # assert self.X.shape[-1] == self.y.shape[-1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # TODO: Improve this
        # X = torch.tensor(self.data[:,:,1:][idx], dtype=torch.float32)
        # y = torch.tensor(self.data[:,:,0][idx], dtype=torch.float32)
        # example = torch.tensor(self.data[idx], dtype=torch.float32)
        self.data
        example = self.data[idx].clone().detach().requires_grad_(True)
        example = example.permute(1, 0)
        example = example.unsqueeze(0).to(torch.float32)
        return example
    
    def computeGlobalStats(self, batchSize=32):
        '''Compute global mean and std using batch processing'''
        loader = DataLoader(self, batch_size=batchSize, shuffle=False)
        print(f'Type: {type(self.data.shape[-1])}, Value: {self.data.shape[-1]}')
        totalSum = torch.zeros(self.data.shape[-1])
        totalSumSq = torch.zeros(self.data.shape[-1])
        totalElements = 0
        
        for batch in loader:
            batchSum = batch.sum(dim=(0,1))
            batchSumSq = (batch ** 2).sum(dim=(0,1))
            totalSum += batchSum
            totalSumSq += batchSumSq
            totalElements += batch.numel() // batch.size(-1)
        
        self.globalMean = totalSum / totalElements
        self.globalStd = torch.sqrt((totalSumSq / totalElements) - self.globalMean ** 2)
            
    def standardizeData(self, save=False):
        '''Standardize data using computed global mean and std'''
        if self.globalMean is None or self.globalStd is None:
            raise ValueError('Global Stats must be computed first')
        
        self.data = (self.data - self.globalMean.numpy()) / self.globalStd.numpy()
        
        if save and self.savePath:
            self.saveStandardizedData()
    
    def saveStandardizedData(self):
        """Save standardized data to an .npz file."""
        np.savez_compressed(
            self.savePath,
            data=self.data,
            mean=self.globalMean.numpy(),
            std=self.globalStd.numpy()
        )
        print(f"Standardized data saved to {self.savePath}")
        
    def loadStandardizedData(self):
        """Load standardized data from an .npz file."""
        if self.standardizedDataPath is None:
            raise ValueError("Path of saved standardized data should be set")

        loaded = np.load(self.standardizedDataPath)
        self.data = torch.tensor(loaded['data'])
        self.globalMean = torch.tensor(loaded['mean'])
        self.globalStd = torch.tensor(loaded['std'])
        self.preprocessed = True
        print(f"Standardized data loaded from {self.standardizedDataPath}")

    def split_data(self, trainRatio=0.8, valRatio=0.1):
        """
        Splits the dataset into training, validation, and test sets based on the given ratios.
        
        Args:
            train_ratio (float): Proportion of the dataset to be used for training.
            val_ratio (float): Proportion of the dataset to be used for validation.
            test_ratio (float): Proportion of the dataset to be used for testing.

        Returns:
            train_dataset, val_dataset, test_dataset (Dataset): Training, validation, and test datasets
        """
        # Check that the sum of ratios is equal to 1
        # assert (trainRatio + valRatio + testRatio) == 1.0, "Ratios must sum to 1."

        # Calculate the number of samples for each subset
        totalSize = len(self)
        trainSize = int(totalSize * trainRatio)
        valSize = int(totalSize * valRatio)
        testSize = totalSize - trainSize - valSize  # The rest for testing

        # Use random_split to split the data into train, validation, and test sets
        trainDataset, valDataset, testDataset = random_split(
            self, [trainSize, valSize, testSize])
        
        return trainDataset, valDataset, testDataset
        
class DatasetFromNPZIter(IterableDataset):
    def __init__(self, npzFile):
        self.data = np.load(npzFile)['noise_event_array']
    
    def __iter__(self):
        for i in range(self.data.shape[0]):
            X = self.data[i,:,1:]
            y = self.data[i,:,0]    

            xTensor = torch.tensor(X, dtype=torch.float32)
            yTensor = torch.tensor(y, dtype=torch.float32)
        
            yield xTensor, yTensor
        
# class TimeSeriesDataset(torch.utils.data.IterableDataset):
#     def __init__(
#             self,
#             X: np.ndarray,
#             y: np.ndarray,
#             batchSize: int,
#             shuffle: bool = True,
#             device: torch.device = "cpu",
#             exampleLength: float = 50
#     ) -> None:
#         assert X.shape[-1] == y.shape[-1]

#         self.batchSize = batchSize
#         self.shuffle = shuffle
#         self.exampleLength = exampleLength

#         X = torch.Tensor(X).to(device)
#         y = torch.Tensor(y).to(device)

#         if self.shuffle:
#             idx = torch.randperm(X.shape[0])
#             X = X[idx]
#             y = y[idx]

class NoiseFinder:
    def __init__(self, Trigger_paths, window_size, pulse_length):
        import uproot
        self.Trigger_paths = Trigger_paths
        self.window_size = window_size
        self.pulse_length = pulse_length
        self.Trigger = np.array([],dtype=int)
        self.posNoise = np.array([],dtype=int)

    def read_Trigger(self):
        self.Trigger = np.array([], dtype=int)
        for Trigger_path in self.Trigger_paths:
            root_file = uproot.open(Trigger_path)
            self.Trigger = np.concatenate(
                (self.Trigger, root_file["trigger_tree"]["TriggerSample"].array(library='np').astype(int)))
        self.Trigger = np.sort(self.Trigger)

    def getNoisePos(self):
        dTrigger = self.Trigger[1:] - self.Trigger[:-1]
        good_interals = dTrigger < self.window_size+self.pulse_length
        self.posNoise = self.Trigger[good_interals] + dTrigger//2

    def save_posNoise(self,path):
        torch_posNoise = torch.from_numpy(self.posNoise)
        torch.save(torch_posNoise,path)


class DatasetFromBinary(Dataset):
    def __init__(self, file_paths, win_length, positions_path=None,
                 globalStd=None, StdCut=None ,channel_out=0, win_shift=0, start_window=None, end_window=None,
                 dtype=np.uint32):
        """
        Args:
            file_paths (list of str): Paths to the binary files.
            positions_path (str): Path to Start positions of the data in the binary file.
            dtype (numpy.dtype): Data type of the binary file.
        """

        self.file_paths = file_paths
        self.positions_path = positions_path
        self.StdCut = StdCut if StdCut is None else torch.tensor(StdCut)
        self.file_length = min([os.path.getsize(file)//np.dtype(dtype).itemsize-1 for file in file_paths])
        if self.positions_path is not None:
            self.positions = torch.load(positions_path).to(torch.int)
            self.positions = self.positions[self.positions<self.file_length-win_length]
        elif StdCut is not None:
            self.positions = torch.arange(0,self.file_length-win_length,win_length)
        else:
            self.positions = torch.arange(0, self.file_length - win_length, win_length)
        self.win_length = win_length
        self.win_shift = int(win_shift)
        self.dtype = dtype
        self.globalStd = globalStd
        self.globalHarmonicStd = globalStd
        self.globalHarmonicStdRMS = torch.zeros(len(self.file_paths))
        self.channel_out = channel_out
        self.start_window = start_window if start_window is not None else win_length // 4
        self.end_window = end_window if end_window is not None else 3 * (win_length // 4)
        # Create a memmap for the binary file
        self.file_maps = [np.memmap(file_path, dtype=dtype, mode='r', offset=4) for file_path in self.file_paths]


    def select_event(self, batchSize=32, HasPulse=False, update_value = False):
        loader = DataLoader(DatasetFromBinary(file_paths=self.file_paths, positions_path=self.positions_path,
                                              win_length=self.win_length, globalStd=self.globalStd, StdCut=self.StdCut,
                                              channel_out=self.channel_out, win_shift=self.win_shift,
                                              start_window=self.start_window, end_window=self.end_window,
                                              dtype=self.dtype), batch_size=batchSize, shuffle=False)
        stds = torch.ones((len(self), len(self.file_paths)))
        for index, batch in loader:
            std = torch.std(batch, dim=2)
            stds[index] = std
            if HasPulse:
                stds[index,self.channel_out] = torch.std(torch.cat((batch[:, self.channel_out, :self.start_window],
                                                                batch[:, self.channel_out, self.end_window:]),
                                                                   dim=1), dim=1)
        sel = torch.all(torch.ge(self.StdCut[None, :], stds),dim=1)
        self.positions = self.positions[sel]
        self.globalStd = torch.mean(stds[sel],dim=0)
        if update_value:
            torch.save(self.positions,self.positions_path[:-3]+"_selected.pt")

    def get_std(self, batchSize=32, select=False):
        '''Compute mean std and Harmonic mean std using batch processing as well as a cut using  Harmonic std'''
        loader = DataLoader(DatasetFromBinary(file_paths=self.file_paths, positions_path=self.positions_path,
                                              win_length=self.win_length,globalStd=self.globalStd,StdCut=self.StdCut,
                                              channel_out=self.channel_out, win_shift=self.win_shift,
                                              start_window=self.start_window, end_window=self.end_window,
                                              dtype=self.dtype), batch_size=batchSize, shuffle=False)
        totalMeanStd = torch.zeros(len(self.file_paths))
        totalInvSumStd = torch.zeros(len(self.file_paths))
        stds = torch.ones((len(self), len(self.file_paths)))
        for index, batch in loader:
            #std = torch.std(torch.cat((batch[:, :, :self.start_window], batch[:, :, self.end_window:]), dim=2), dim=2)
            std = torch.std(batch, dim=2)
            if select:
                stds[index] = std
            batchMeanStd = torch.mean(std,dim=0)
            totalInvSumStd += torch.sum(1/std,dim=0)
            totalMeanStd += batchMeanStd
        self.globalStd = totalMeanStd / len(loader)
        self.globalHarmonicStd = torch.reciprocal(totalInvSumStd) * len(self)
        if select:
            self.globalHarmonicStdRMS = torch.sqrt(torch.sum((1 / stds - 1 / self.globalHarmonicStd[None, :]) ** 2, dim=0)/(len(self)-1))*self.globalHarmonicStd**2
            sel = torch.all(torch.ge(self.globalHarmonicStd[None, :] + 3 * self.globalHarmonicStdRMS[None, :], stds),
                            dim=1)
            self.positions = self.positions[sel]
            file_name, file_extension = os.path.splitext(self.positions_path)
            torch.save(self.positions, file_name + "_selected" + file_extension)
            self.positions_path = file_name + "_selected" + file_extension
            self.get_std(batchSize,False)

    def exclude_positions(self, exclude_pos_path):#could be improved by reducing the window rejection
        exclude_positions = torch.load(exclude_pos_path).to(torch.int)
        right_idx = torch.searchsorted(exclude_positions, self.positions, right=True)
        right_idx_clamped = torch.clamp(right_idx, 0, len(exclude_positions) - 1)
        # Get neighboring values: left (before insert) and right (after insert)
        left_neighbors = torch.where(right_idx > 0, exclude_positions[right_idx - 1], float('-inf'))
        right_neighbors = torch.where(right_idx < len(exclude_positions), exclude_positions[right_idx_clamped], float('inf'))
        # Mask out elements too close to neighbors
        keep_mask = ((torch.abs(self.positions - left_neighbors) >= self.win_length) &
                     (torch.abs(self.positions - right_neighbors) >= self.win_length))
        self.positions = self.positions[keep_mask]


    def load_and_process_window_torch(self,file_map, start):
        np_data = np.array(file_map[start:start + self.win_length], dtype=self.dtype)
        np_data = ((np.right_shift(np_data, 8).astype(np.float32) / (2 ** 23)) * 10.24 - 10.24).astype(np.float32)
        tensor_data = torch.from_numpy(np_data)  # Convert to PyTorch
        return tensor_data

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        start = self.positions[index] - self.win_shift
        data = torch.stack([self.load_and_process_window_torch(file_map, start) for file_map in self.file_maps])
        return index, data



class DatasetSTFTFromBinary(DatasetFromBinary):
    def __init__(self, file_paths, win_length, positions_path=None,
                 globalStd=None, StdCut=None,
                 n_fft=128, hop_length=64, channel_out=0, win_shift=0, start_window=None, end_window=None,
                 select_sample=False,dtype=np.uint32):
        super().__init__(file_paths, win_length, positions_path=positions_path,globalStd=globalStd, StdCut=StdCut ,
                         channel_out=channel_out, win_shift=win_shift, start_window=start_window, end_window=end_window,
                         dtype=dtype)
        if globalStd is None:
            if positions_path is None:
                self.select_event()
            else:
                self.get_std(32, select_sample)
        elif isinstance(globalStd, list):
            self.globalStd = torch.tensor(globalStd)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fftwin_length = n_fft
        self.window = torch.hann_window(self.fftwin_length)

    def __getitem__(self, index):
        # Get the trace from base class
        index,data = super().__getitem__(index)
        # Compute the baseline
        mean = torch.mean(torch.cat((data[:, :self.start_window], data[:, self.end_window:]), dim=1), dim=1)
        # Remove the baseline and Standardize
        data = (data - mean[:, None]) / self.globalStd[:, None]
        # Clone the window to predict
        y = torch.clone(data[self.channel_out, self.start_window:self.end_window])
        # Blind the output in the input
        data[self.channel_out, self.start_window:self.end_window] = 0
        # Compute the STFT
        stft = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.fftwin_length,
                          onesided=True,center=True, return_complex=False)
        B,N,T,C = stft.shape
        # Reshape to Channels,Frames,Frequencies
        stft = stft.permute(3,0, 2, 1).reshape(B*C, T, N)
        return stft, y

class DatasetSTFTFromNpy(Dataset):
    def __init__(self, file_path, win_length,
                 n_fft, hop_length, channel_out=0, start_window=None, end_window=None):
        self.data = torch.from_numpy(np.transpose(np.load(file_path, mmap_mode='r'), (1,0,2))[:10000].astype(np.float32)).float()
        mean = torch.mean(torch.cat((self.data [:, :, :start_window], self.data [:, :, end_window:]), dim=2), dim=2)
        std = torch.std(torch.cat((self.data[:, :, :start_window], self.data[:, :, end_window:]), dim=2), dim=2)
        self.data = (self.data - mean[:, :, None]) / torch.mean(std, dim=0)[None,:,None]
        # extract the data to predict
        self.y = torch.clone(self.data[:, channel_out, start_window:end_window])
        # set the data to predict to 0 in the input
        self.data[:, channel_out, start_window:end_window] = 0
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fftwin_length = n_fft
        self.window = torch.hann_window(self.fftwin_length)

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        stft = torch.stft(self.data[index], n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.fftwin_length, onesided=True,center=True,
                          return_complex=False)
        B,N,T,C = stft.shape
        stft = stft.permute(3,0, 2, 1).reshape(B*C,T,N)
        return stft,self.y[index]