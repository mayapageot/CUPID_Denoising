import numpy as np
import torch

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def Fit(self, data):
        """
        Calculate and store the mean and standard deviation of the dataset.
        
        Parameters:
        data (np.ndarray or torch.Tensor): The dataset to calculate mean and std from, with shape (n_samples, n_features).
        """
        if isinstance(data, torch.Tensor):
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:  # Assuming data is a numpy array
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        # TODO: throw error if data is neither torch tensor nor numpy array

    def Transform(self, data):
        """
        Standardize the dataset using the stored mean and std.
        
        Parameters:
        data (np.ndarray or torch.Tensor): The dataset to transform, with shape (n_samples, n_features).
        
        Returns:
        np.ndarray or torch.Tensor: The standardized dataset.
        """
        if self.mean is None or self.std is None:
            raise ValueError("The Standardizer instance is not fitted yet. Call 'fit' with training data first.")
        
        return (data - self.mean) / self.std

    def InverseTransform(self, data):
        """
        Scale up the dataset using the stored mean and std.
        
        Parameters:
        data (np.ndarray or torch.Tensor): The standardized dataset to scale up, with shape (n_samples, n_features).
        
        Returns:
        np.ndarray or torch.Tensor: The scaled-up dataset.
        """
        if self.mean is None or self.std is None:
            raise ValueError("The Standardizer instance is not fitted yet. Call 'fit' with training data first.")
        
        return data * self.std + self.mean

    def FitTransform(self, data):
        """
        Fit to the data, then transform it.
        
        Parameters:
        data (np.ndarray or torch.Tensor): The dataset to fit and transform, with shape (n_samples, n_features).
        
        Returns:
        np.ndarray or torch.Tensor: The standardized dataset.
        """
        self.Fit(data)
        return self.Transform(data)