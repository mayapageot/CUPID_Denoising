import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

class DatasetFromNPZ(Dataset):
    def __init__(self, 
                 npzFile,
                 batchSize: int):
        data = np.load(npzFile)
        self.data = data['noise_event_array']
        self.X = self.data[:,:,1:] # witness channel output
        self.y = self.data[:,:,0] # bolometer output

        # TODO: Assert that each example is of the same length
        #assert X.shape[-1] == y.shape[-1]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X,y
        
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
        
        