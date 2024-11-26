import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional

LRELU_SLOPE = 0.1

class MRFLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                channels, channels, kernel_size, padding=kernel_size // 2, dilation=1
            )
        )

    def forward(self, x):
        y = F.leaky_relu(x, LRELU_SLOPE)
        y = self.conv1(y)
        y = F.leaky_relu(y, LRELU_SLOPE)
        y = self.conv2(y)
        return x + y

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)


class MRFBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(MRFLayer(channels, kernel_size, dilation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()

class SineGen(torch.nn.Module):

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sample_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0, upp):
        a = torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        rad = f0 / self.sample_rate * a
        rad2 = torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += torch.nn.functional.pad(rad_acc, (0, 0, 1, 0), mode='constant')
        rad = rad.reshape(f0.shape[0], -1, 1)
        b = torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        rad *= b
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad)
        return sines

    def forward(self, f0: torch.Tensor, upp: int):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._f02sine(f0, upp) * self.sine_amp
            uv = self._f02uv(f0)
            uv = torch.nn.functional.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

class SourceModuleHnNSF(torch.nn.Module):
    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upsample_factor):
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        return sine_merge, None, None

class HiFiGAN(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
        gin_channels,
        sample_rate,
        harmonic_num,
    ):
        super().__init__()
        print('New')
        self.num_kernels = len(resblock_kernel_sizes)
        
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
            )
        )
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=u // 2 + u % 2,
                        output_padding=u % 2,
                    )
                )
            )
            if i < len(upsample_rates) - 1:
                stride_f0 = np.prod(upsample_rates[i + 1 :])  # noqa
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(
                    nn.Conv1d(
                        1, upsample_initial_channel // (2 ** (i + 1)), kernel_size=1
                    )
                )

        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                nn.ModuleList(
                    [
                        MRFBlock(channel, kernel_size=k, dilations=d)
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )
        self.conv_post = weight_norm(
            nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3)
        )
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        self.upp = math.prod(upsample_rates)

    def forward(self, x, f0, g: Optional[torch.Tensor] = None):
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(-1, -2)
        
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)        
        
        for up, mrf, noise_conv in zip(self.upsamples, self.mrfs, self.noise_convs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x_source = noise_conv(har_source)
            x = x + x_source
            xs = 0
            for layer in mrf:
                xs += layer(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)