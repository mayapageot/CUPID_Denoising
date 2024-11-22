import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional

#set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvBlock(nn.Module):
    def __init__(
            self,
            inChannels: int,
            outChannels: int,
            kernelSize: tuple[int, int],
            stride: int,
            padding: int,
            transpose: bool,
            outputPadding: Optional[int] = None,
            activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()

        if not transpose and outputPadding is not None:
            raise ValueError(
                "Output padding only valid for transpose convoltions!"
            )
        elif outputPadding is not None:
            kwargs = {"output_padding": outputPadding}
        else:
            kwargs = {}
        
        convLayer = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = convLayer(
            inChannels,
            outChannels,
            kernel_size=kernelSize,
            stride=stride,
            padding=padding,
            **kwargs,
        )
        self.bn = nn.BatchNorm2d(outChannels)
        self.activation = activation()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ModelAE(nn.Module):
    def __init__(self, inChannels, kernelLength=10, corrWitness=2):
        super().__init__()

        self.numWitnesses = inChannels
        self.inputConv = ConvBlock(
            inChannels,
            inChannels,
            kernelSize=(corrWitness,kernelLength),
            stride=1,
            padding='same',
            transpose=False,
        )

        self.downsampler = nn.Sequential()
        for i, outChannels in enumerate([2, 4, 8, 16]):
            convBlock = ConvBlock(
                inChannels,
                outChannels,
                kernelSize=(corrWitness,kernelLength),
                stride=2,
                padding=3,
                transpose=False,
            )
            self.downsampler.add_module(f'CONV_{i+1}', convBlock)
            inChannels = outChannels

        self.upsampler = nn.Sequential()
        for i,outChannels in enumerate([32,16,8, self.numWitnesses]):
            convBlock = ConvBlock(
                    inChannels,
                    outChannels,
                    kernelSize=(corrWitness,kernelLength),
                    stride=2,
                    padding=3,
                    outputPadding=1,
                    transpose=True,
            )
            self.upsampler.add_module(f'TRANS_CONV_{i+1}', convBlock)
            inChannels = outChannels
        
        self.outputConv = nn.Conv2d(
            inChannels, 1, kernel_size=(corrWitness,kernelLength), stride=1, padding='same'
        )

    def forward(self, x):
        x = self.inputConv(x)
        x = self.downsampler(x)
        x = self.upsampler(x)
        x = self.outputConv(x)
        return x[:, 0]