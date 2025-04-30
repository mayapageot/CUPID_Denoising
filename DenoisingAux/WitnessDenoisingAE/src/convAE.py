import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from functools import reduce

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvBlock(nn.Module):
    def __init__(
            self,
            inChannels: int,
            outChannels: int,
            kernelSize: tuple[int, int],
            stride,
            padding,
            transpose: bool,
            outputPadding: Optional[tuple[int, int]] = None,
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
    def __init__(self, inChannels, numWitnesses, kernelLength=10, corrWitness=2):
        super().__init__()

        self.numWitnesses = numWitnesses
        self.inputConv = ConvBlock(
            inChannels,
            inChannels,
            kernelSize=(corrWitness, kernelLength),
            stride=1,
            padding='same',
            transpose=False,
        )

        self.downsampler = nn.Sequential()
        for i, outChannels in enumerate([2, 4, 8, 16]):
            convBlock = ConvBlock(
                inChannels,
                outChannels,
                kernelSize=(corrWitness, kernelLength),
                stride=(1, 2),
                padding=(0, kernelLength // 2),
                transpose=False,
            )
            self.downsampler.add_module(f'CONV_{i + 1}', convBlock)
            inChannels = outChannels

        self.upsampler = nn.Sequential()
        for i, outChannels in enumerate([16, 8, 4, 1]):
            if (outChannels != 1):
                convBlock = ConvBlock(
                    inChannels,
                    outChannels,
                    kernelSize=(corrWitness, kernelLength),
                    stride=(1, 2),
                    padding=(0, kernelLength // 2),
                    outputPadding=(0, 1),
                    transpose=True,
                )
            else:
                convBlock = ConvBlock(
                    inChannels,
                    outChannels,
                    kernelSize=(corrWitness, kernelLength),
                    stride=(1, 2),
                    padding=(0, kernelLength // 2),
                    outputPadding=(0, 0),
                    transpose=True,
                )
            self.upsampler.add_module(f'TRANS_CONV_{i + 1}', convBlock)
            inChannels = outChannels

        self.outputConv1 = nn.Conv2d(
            inChannels,
            1,
            kernel_size=(numWitnesses // 2, 1),
            stride=1,
            padding=0
        )

        self.outputConv2 = nn.Conv2d(
            1,
            1,
            kernel_size=((numWitnesses // 2) + 2, 1),
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.inputConv(x)
        x = self.downsampler(x)
        x = self.upsampler(x)
        x = self.outputConv1(x)
        x = self.outputConv2(x)
        return x


class GroupedLSTM(nn.Module):
    def __init__(self, group_number, input_size, activation=torch.nn.Tanh):
        super(GroupedLSTM, self).__init__()
        self.group_number = group_number
        self.input_size = input_size
        # Calculate sub-units for each group
        self.sub_input_sizes = [
            self.input_size // self.group_number + (1 if i < self.input_size % self.group_number else 0)
            for i in range(self.group_number)]
        # Initialize separate LSTM layers for each group
        self.subLSTMs = nn.ModuleList([nn.LSTM(input_size=sub_input_size, hidden_size=sub_input_size, batch_first=True)
                                       for sub_input_size in self.sub_input_sizes])
        self.activation = activation()

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.shape
        xs = torch.split(inputs, self.sub_input_sizes, dim=-1)
        xs = [subLSTM(x)[0] for subLSTM, x in zip(self.subLSTMs, xs)]
        stacked = torch.cat(xs, dim=-1)
        x = stacked.flatten(start_dim=2)
        # if self.activation is not None:
        #     x = self.activation(x)
        return x


class DeConv2D_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0):
        super(DeConv2D_skip, self).__init__()
        self.Conv2DTranspose = nn.ConvTranspose2d(in_channels=in_channels * 2, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, inputs, skip_inputs):
        # Calculate the padding required for height and width
        pad_height = skip_inputs.size(2) - inputs.size(2)
        pad_width = skip_inputs.size(3) - inputs.size(3)

        # Ensure valid padding
        if pad_height < 0 or pad_width < 0:
            raise ValueError(f"Negative padding: pad_height={pad_height}, pad_width={pad_width}")

        # Pad inputs
        inputs_padded = F.pad(inputs, (0, pad_width, 0, pad_height))
        x = torch.cat((skip_inputs, inputs_padded), dim=1)
        x = self.Conv2DTranspose(x)
        return x


class AutoencoderSkipLayers(nn.Module):
    def __init__(self, filters, kernel_size, strides,
                 hop_length, win_length, n_fft, out_window_size, n_in, n_out, activation=nn.ELU):
        super(AutoencoderSkipLayers, self).__init__()
        self.activation = activation()
        self.n_out = n_out
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.n_fft = n_fft
        self.fftwin_length = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        window = torch.hann_window(self.fftwin_length)
        self.register_buffer("window", window)
        self.unknown_window_size = out_window_size
        self.num_frame = self.win_length // self.hop_length + 1
        self.num_freq = self.n_fft // 2 + 1
        self.Input_shape = (n_in, self.num_frame, self.num_freq)
        self.Conv2D_layers = nn.ModuleList([
            nn.Conv2d(in_channels=n_in if i == 0 else filters[i - 1],
                      out_channels=f, kernel_size=kernel_size, stride=strides)
            for i, f in enumerate(filters)
        ])
        latent_space_size = reduce(lambda x, _: (x - 1) // 2, range(len(filters)), self.num_freq)
        print("latent_space_size: ", latent_space_size)
        self.DeConv2D_layers = nn.ModuleList([DeConv2D_skip(in_channels=filters[i - 1],
                                                            out_channels=filters[i - 2] if i != 1 else 2,
                                                            kernel_size=(1, 3), stride=(1, 2))
                                              for i in range(len(filters), 0, -1)])

    def forward(self, x):
        skip_layers = []
        for Conv2D in self.Conv2D_layers:
            x = Conv2D(x)
            if self.activation is not None:
                x = self.activation(x)
            skip_layers.append(x)
        for i, (ConvTranspose2D, skip_layer) in enumerate(zip(self.DeConv2D_layers, skip_layers[::-1])):
            x = ConvTranspose2D(x, skip_layer)
            # x = ConvTranspose2D(x)
            if self.activation is not None and i < len(self.DeConv2D_layers) - 1:
                x = self.activation(x)
        pad = self.num_freq - x.shape[3]
        x = torch.nn.functional.pad(x, (0, pad))
        x = torch.complex(x[:, 0], x[:, 1])
        x = torch.istft(x.permute(0, 2, 1), n_fft=self.n_fft, hop_length=self.hop_length,
                        win_length=self.fftwin_length, window=self.window, onesided=True, length=self.win_length)
        x = x[:, (self.win_length - self.unknown_window_size) // 2:(self.win_length + self.unknown_window_size) // 2]

        return x


class AutoencoderLSTM(nn.Module):
    def __init__(self, filters, kernel_size, strides, n_LSTM, LSTM_group_number,
                 hop_length, win_length, n_fft, out_window_size, n_in, n_out,
                 unknown_window_center=0.5, activation=nn.ELU):
        super(AutoencoderLSTM, self).__init__()
        self.activation = activation()
        self.n_out = n_out
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.n_LSTM = n_LSTM
        self.LSTM_group_number = LSTM_group_number
        self.n_fft = n_fft
        self.fftwin_length = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        window = torch.hann_window(self.fftwin_length)
        self.register_buffer("window", window)
        self.unknown_window_size = out_window_size
        self.unknown_window_center = unknown_window_center
        self.num_frame = torch.stft(torch.zeros(self.win_length), n_fft=n_fft, hop_length=hop_length,win_length=n_fft,
                                    onesided=True,center=True,return_complex=False).shape[1]
        #self.num_frame = self.win_length // self.hop_length + 1
        self.num_freq = self.n_fft // 2 + 1
        self.Input_shape = (n_in, self.num_frame, self.num_freq)
        self.Conv2D_layers = nn.ModuleList([
            nn.Conv2d(in_channels=n_in if i == 0 else filters[i - 1],
                      out_channels=f, kernel_size=kernel_size, stride=strides)
            for i, f in enumerate(filters)
        ])
        self.latent_space_size = reduce(lambda x, _: (x - 1) // 2, range(len(filters)), self.num_freq)
        self.GrLSTMs = nn.ModuleList([GroupedLSTM(group_number=LSTM_group_number, input_size=filters[-1] * self.latent_space_size) for _ in range(self.n_LSTM)])
        #self.GrLSTM2 = GroupedLSTM(group_number=LSTM_group_number, input_size=filters[-1] * latent_space_size)
        self.reshapeLayer = [self.num_frame, filters[-1], self.latent_space_size]
        self.DeConv2D_layers = nn.ModuleList([DeConv2D_skip(in_channels=filters[i - 1],
                                                            out_channels=filters[i - 2] if i != 1 else 2,
                                                            kernel_size=(1, 3), stride=(1, 2))
                                              for i in range(len(filters), 0, -1)])

    def forward(self, x):
        skip_layers = []
        for Conv2D in self.Conv2D_layers:
            x = Conv2D(x)
            if self.activation is not None:
                x = self.activation(x)
            skip_layers.append(x)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        for GrLSTM in self.GrLSTMs :
            x = GrLSTM(x)
        # x = self.GrLSTM1(x)
        # x = self.GrLSTM2(x)
        x = x.reshape(-1, *self.reshapeLayer).permute(0, 2, 1, 3)
        for i, (ConvTranspose2D, skip_layer) in enumerate(zip(self.DeConv2D_layers, skip_layers[::-1])):
            x = ConvTranspose2D(x, skip_layer)
            if self.activation is not None and i < len(self.DeConv2D_layers) - 1:
                x = self.activation(x)
        pad = self.num_freq - x.shape[3]
        x = torch.nn.functional.pad(x, (0, pad))
        x = torch.complex(x[:, 0], x[:, 1])
        x = torch.istft(x.permute(0, 2, 1), n_fft=self.n_fft, hop_length=self.hop_length,
                        win_length=self.fftwin_length, window=self.window, onesided=True, length=self.win_length)
        x = x[:, int(self.win_length * self.unknown_window_center) - self.unknown_window_size // 2:
                 int(self.win_length * self.unknown_window_center) - (self.unknown_window_size // -2)]
        return x


class AutoencoderLSTM_restricted(AutoencoderLSTM):
    def __init__(self, filters, kernel_size, strides, n_LSTM, LSTM_group_number,
                 n_fft_base, n_fft_multiplier, hop_length_frac_power, win_length, out_window_size, n_in, n_out,
                 unknown_window_center=0.5, activation=nn.ELU):
        self.n_fft_base=n_fft_base
        self.n_fft_multiplier = n_fft_multiplier
        self.hop_length_frac_power = hop_length_frac_power
        n_fft = self.n_fft_base*self.n_fft_multiplier
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides,
                         n_LSTM=n_LSTM, LSTM_group_number=LSTM_group_number,
                         hop_length=n_fft/self.hop_length_frac_power, win_length=win_length, n_fft=n_fft,
                         out_window_size=out_window_size, n_in=n_in, n_out=n_out,
                         unknown_window_center=unknown_window_center, activation=activation)