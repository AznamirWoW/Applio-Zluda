import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d, Conv1d
from torch.nn.utils import weight_norm, spectral_norm
from torch import nn
from rvc.lib.algorithm.vocoder_blocks import *

LRELU_SLOPE = 0.1

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm        
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(   1,  32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0),)),
                norm_f(Conv2d(  32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0),)),
                norm_f(Conv2d( 128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0),)),
                norm_f(Conv2d( 512,1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0),)),
                norm_f(Conv2d(1024,1024, (kernel_size, 1), (stride, 1), padding=(2, 0),)),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU_SLOPE, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(
            self,
            periods=[2, 3, 5, 7, 11, 17, 23, 37]
        ):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(p) for p in periods]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
