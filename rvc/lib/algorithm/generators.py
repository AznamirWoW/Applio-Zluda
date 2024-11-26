import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional

from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock1, ResBlock2
from rvc.lib.algorithm.commons import init_weights

import numpy as np

class Generator(torch.nn.Module):
    """Generator for synthesizing audio. Optimized for performance and quality.

    Args:
        initial_channel (int): Number of channels in the initial convolutional layer.
        resblock (str): Type of residual block to use (1 or 2).
        resblock_kernel_sizes (list): Kernel sizes of the residual blocks.
        resblock_dilation_sizes (list): Dilation rates of the residual blocks.
        upsample_rates (list): Upsampling rates.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes of the upsampling layers.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 0.
    """

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs == None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def __prepare_scriptable__(self):
        """Prepares the module for scripting."""
        for l in self.ups_and_resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self

    def remove_weight_norm(self):
        """Removes weight normalization from the upsampling and residual blocks."""
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SineGen(torch.nn.Module):
    """Sine wave generator.

    Args:
        samp_rate (int): Sampling rate in Hz.
        harmonic_num (int, optional): Number of harmonic overtones. Defaults to 0.
        sine_amp (float, optional): Amplitude of sine waveform. Defaults to 0.1.
        noise_std (float, optional): Standard deviation of Gaussian noise. Defaults to 0.003.
        voiced_threshold (float, optional): F0 threshold for voiced/unvoiced classification. Defaults to 0.
        flag_for_pulse (bool, optional): Whether this SineGen is used inside PulseGen. Defaults to False.
    """

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
        """Converts F0 to voiced/unvoiced signal.

        Args:
            f0 (torch.Tensor): F0 tensor with shape (batch_size, length, 1)..
        """
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0, upp):
        """ f0: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
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
        """Generates sine waves.

        Args:
            f0 (torch.Tensor): F0 tensor with shape (batch_size, length, 1).
            upp (int): Upsampling factor.
        """
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
