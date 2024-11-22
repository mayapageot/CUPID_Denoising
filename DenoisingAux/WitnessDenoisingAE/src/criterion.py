from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.signal.spectral import _median_bias

def _validateShapes(x: torch.Tensor, y: torch.Tensor):
    # acceptable combinations of shapes:
    # x: time, y: time
    # x: channel x time, y: time OR channel x time
    # x: batch x channel x time, y: batch x channel x time OR batch x time
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(
            "Time dimensions of x and y tensors must "
            "be the same, found {} and {}".format(x.shape[-1], y.shape[-1])
        )
    elif x.ndim == 1 and not y.ndim == 1:
        raise ValueError(
            "Can't compute cross spectral density of "
            "1D tensor x with {}D tensor y".format(y.ndim)
        )
    elif x.ndim > 1 and y.ndim == x.ndim:
        if not y.shape == x.shape:
            raise ValueError(
                "If x and y tensors have the same number "
                "of dimensions, shapes must fully match. "
                "Found shapes {} and {}".format(x.shape, y.shape)
            )
    elif x.ndim > 1 and y.ndim != (x.ndim - 1):
        raise ValueError(
            "Can't compute cross spectral density of "
            "tensors with shapes {} and {}".format(x.shape, y.shape)
        )
    elif x.ndim > 2 and y.shape[0] != x.shape[0]:
        raise ValueError(
            "If x is a 3D tensor and y is a 2D tensor, "
            "0th batch dimensions must match, but found "
            "values {} and {}".format(x.shape[0], y.shape[0])
        )


def _median(x, axis):
    return torch.quantile(x, q=0.5, axis=axis)

def _fastCSD(
    x: torch.Tensor,
    nExample: int,
    nStride: int,
    window: torch.Tensor,
    scale: torch.Tensor,
    average: str,
    y: Optional[torch.Tensor] = None,
):
    # for fast implementation, use the built-in stft
    # module along the last dimension and let torch
    # handle the heavy lifting under the hood
    if x.ndim > 2:
        # stft only works on 2D input, so roll the
        # channel dimension out along the batch
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        x = x.reshape(-1, x.shape[-1])

        if y is not None and y.ndim == 3:
            y = y.reshape(-1, x.shape[-1])
    else:
        batch_size = None

    x = x - x.mean(axis=-1, keepdims=True)
    fft = torch.stft(
        x,
        n_fft=nExample,
        hop_length=nStride,
        window=window,
        normalized=False,
        center=False,
        return_complex=True,
    )
    if y is not None:
        y = y - y.mean(axis=-1, keepdims=True)
        y_fft = torch.stft(
            y,
            n_fft=nExample,
            hop_length=nStride,
            window=window,
            normalized=False,
            center=False,
            return_complex=True,
        )
        if batch_size is not None and fft.shape[0] > y_fft.shape[0]:
            nfreq = nExample // 2 + 1
            fft = fft.reshape(batch_size, num_channels, nfreq, -1)
            y_fft = y_fft[:, None]

        fft = torch.conj(fft) * y_fft
    else:
        fft = fft.abs() ** 2

    stop = None if nExample % 2 else -1
    if x.ndim == 1:
        fft[1:stop] *= 2
    elif fft.ndim < 4:
        fft[:, 1:stop] *= 2
    else:
        fft[:, :, 1:stop] *= 2
    fft *= scale

    if average == "mean":
        fft = fft.mean(axis=-1)
    else:
        bias = _median_bias(fft.shape[-1])
        if y is not None:
            real_median = _median(fft.real, -1)
            imag_median = 1j * _median(fft.imag, -1)
            fft = real_median + imag_median
        else:
            fft = _median(fft, -1)
        fft /= bias

    if fft.ndim == 2 and batch_size is not None:
        fft = fft.reshape(batch_size, num_channels, -1)
    return fft

class TorchWelch(nn.Module):
    def __init__(
            self,
            sampleRate: float,
            fftLength: float,
            overlap: Optional[float] = None,
            average: str = "mean",
            device: str = "cpu",
            fast: bool = False,
    ):
        if overlap is None:
            overlap = fftLength / 2
        elif overlap >= fftLength:
            raise ValueError(
                f'Overlap {overlap} cannot be longer than fftLength {fftLength}'
                )
        super().__init__()

        # number of samples in a training example
        self.nExample = int(fftLength * sampleRate)

        # stride of fft window
        self.nStride = self.nExample - int(overlap * sampleRate)

        # hann window to perform DFT
        self.window = torch.hann_window(self.nExample).to(device)

        # normalization over the window
        self.scale = 1.0 / (sampleRate * (self.window**2).sum())

        if average not in ("mean", "median"):
            raise ValueError(
                f'Average must be "mean" or "median", input value was {average}'
            )

        self.average = average
        self.device = device

    def forward(self, x:torch.Tensor, y: Optional[torch.Tensor] = None):
        if x.shape[-1] < self.nExample:
            raise ValueError(
                f'Number of samples {x.shape[-1]} in input x is insufficient for
                number of fft samples {self.nExample}'
            )
        elif x.ndim > 3:
            raise ValueError(
                f'Cannot perform welch transform on tensor with shape {x.shape}'
            )
        
        if y is not None:
            _validateShapes(x,y)

        if x.ndim == 1:
            reshape = []
            x = x[None, None, None, :]
        elif x.ndim == 2:
            reshape = [len(x)]
            x = x[None, :, None, :]
        elif x.ndim == 3:
            reshape = list(x.shape[:-1])
            x = x[:, :, None, :]

        # calculate the number of segments and trim x along
        # the time dimensions so that we can unfold it exactly
        numSegments = ((x.shape[-1] - self.nExample) // self.nStride) + 1
        end = (numSegments - 1) * self.nStride + self.nExample
        x = x[..., :end]

        # unfold x into overlapping segments and detrend and window
        # each one individually before computing the rfft. Unfold
        # will produce a batch x (num_channels * num_segments) x nExample
        # shaped tensor
        unfold_op = torch.nn.Unfold(
            (1, numSegments), dilation=(1, self.nStride)
        )
        x = unfold_op(x)
        x = x - x.mean(axis=-1, keepdims=True)
        x *= self.window

        # after the fft, we'll have a
        # batch x (num_channels * num_segments) x nfreq
        # sized tensor
        fft = torch.fft.rfft(x, axis=-1).abs() ** 2

        if self.nExample % 2:
            fft[..., 1:] *= 2
        else:
            fft[..., 1:-1] *= 2
        fft *= self.scale

        # unfold the batch and channel dimensions back
        # out if there were any to begin with, putting
        # the segment dimension as the second to last
        reshape += [numSegments, -1]
        fft = fft.reshape(*reshape)

        if self.average == "mean":
            return fft.mean(axis=-2)
        else:
            bias = _median_bias(numSegments)
            return _median(fft, -2) / bias

class PSDLoss(nn.Module):
    '''Calculate PSD loss (average over frequency of the PSD ratio)'''
    def __init__(
            self,
            sampleRate: float,
            fftLength: float,
            overlap: float,
            freqLow: Union[float, List[float], None] = None,
            freqHigh: Union[float, List[float], None] = None,
            device: str = "cpu",
    ):
        super().__init__()

        if freqLow is not None and freqHigh is not None:
           # remove frequencies outside specified range
           dfreq = 1 / fftLength
           freqs = np.arange(0.0, sampleRate / 2 + dfreq, dfreq)
           mask = np.zeros_like(freqs, dtype=bool)
           mask = (freqLow <= freqs) & (freqs < freqHigh)

        elif freqLow is None and freqHigh is None:
            self.mask = None

        else:
            # raise error if one was specified and one was not
            raise ValueError(
                "If {} is specified, '{} must be specified as well".format(
                    "freqHigh" if freqLow is None else "freqLow",
                    "freqLow" if freqLow is None else "freqHigh",
                )
            )
        
        self.welch = TorchWelch(
            sampleRate,
            fftLength,
            overlap,
            average="median",
            device=device, 
        )
    
    def forward(self, pred, target):
        # Calculate the PSD of the residual and the bolometer channel, and return the mean of their ratio
        psdRes = self.welch(target - pred)
        psdTarget = self.welch(target)

        if self.mask is not None:
            psdRes = torch.masked_select(psdRes, self.mask)
            psdRes = psdRes.reshape(len(pred),-1)

            psdTarget = torch.masked_select(psdTarget, self.mask)
            psdTarget = psdTarget.reshape(len(pred), -1)

        ratio = psdRes / psdTarget
        return ratio.mean()
    
