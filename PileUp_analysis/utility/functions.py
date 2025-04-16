import numpy as np
import torch
def modified_gaussian(x,A,x0,sigma,b):
    y = np.ones_like(x)
    y[x>0] = A*np.exp(-(np.log(x[x>0])-x0)**2/(2*sigma**2))+b
    y[x<0] =  A*np.exp(-(np.log(-x[x<0])-x0)**2/(2*sigma**2))+b
    return y

def polyfit_batch(x, y, degree):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    # x, y shape: (batch, n)
    X = torch.stack([x ** i for i in reversed(range(degree + 1))], dim=-1)  # (batch, n, degree+1)
    XT = X.transpose(1, 2)  # (batch, degree+1, n)

    # Normal equation: (X^T X)^(-1) X^T y
    coeffs = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)  # (batch, degree+1)
    return coeffs