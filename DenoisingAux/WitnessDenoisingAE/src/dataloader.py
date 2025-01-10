import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import IterableDataset

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


class DatasetFromBinary(Dataset):
    def __init__(self, file_paths, positions, length, dtype=np.float32):
        """
        Args:
            file_paths (list of str): Paths to the binary files.
            positions (list of int): Start positions of the data in the binary file.
            ranges (list of int): Length of data to read from each position.
            dtype (numpy.dtype): Data type of the binary file.
        """
        self.file_paths = file_paths
        self.positions = positions
        self.length = length
        self.dtype = dtype

        # Create a memmap for the binary file
        self.file_maps = [np.memmap(file_path, dtype=dtype, mode='r') for file_path in self.file_paths]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        start = self.positions[index]
        # Extract the data using slicing
        data = np.array([file_map[start:start + self.length].astype(np.float32) for file_map in self.file_maps])
        return torch.tensor(data, dtype=torch.float32)