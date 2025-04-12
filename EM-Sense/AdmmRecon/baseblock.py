import torch
import torch.nn as nn
import numpy as np
from typing import NamedTuple



class ResnetConfig(NamedTuple):
    in_channels:int = 2
    hidden_channels:int = 64
    out_channels:int = 2
    iters:int = 5
    act:nn.Module = nn.SiLU()
    attention:bool = True
    C: int = 16 # expend channels



def conv(
        in_channels:int,
        out_channels:int,
        kernel_size:int=3,
        groups:int=1,
        bias:bool=False,
        stride:int=1,
):
    return nn.Conv2d(in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size -1) // 2,
                groups=groups,
                stride=stride,
                bias=bias)


class MBconv(nn.Module):
    def __init__(self, 
                in_channels,
                out_channels,
                kernel_size=3,
                C = 16,
                act=nn.SiLU(),
                ):
        super().__init__()

        self.conv = conv(in_channels=in_channels, out_channels=in_channels + C, kernel_size=kernel_size)
        self.act = act
        self.pwconv = conv(in_channels=in_channels + C, out_channels=out_channels, kernel_size=1)
    def forward(self, x):
        
        out = self.conv(x)
        out = self.act(out)
        out = self.pwconv(out) + x
        return out


class ConvNeXt(nn.Module):
    def __init__(self, 
                in_channels,
                out_channels,
                act=nn.SiLU(),
                attention=True,
                ):
        super().__init__()

        self.dwconv = conv(in_channels=in_channels, out_channels=in_channels, kernel_size=7, groups=in_channels)
        self.attention = simam_module() if attention else nn.Identity()
        self.act1 = act
        self.pwconv1= conv(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1)
        self.act1 = act
       
        self.pwconv2 = conv(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
    def forward(self, x):
        out = self.dwconv(x)
        out = self.attention(out)
        out = self.pwconv1(out)
        out = self.act1(out)
        out = self.pwconv2(out) + x

        return out

class ReconBlock(nn.Module):
    def __init__(self, 
                channels,
                act=nn.SiLU(),
                C:int=16,
                attention=True,
                ):
        super().__init__()

        self.convnext1 = ConvNeXt(in_channels=channels, out_channels=channels, act=act, attention=attention)
        self.mbconv = MBconv(in_channels=channels, out_channels=channels, act=act, C=C)
    def forward(self, x):
        
        out = self.convnext1(x)

        out = self.mbconv(out)

        return out
        
    
class resnet(nn.Module):
    def __init__(self,
                Resnetconfig=ResnetConfig,
                ):
        super().__init__()
        in_channels = Resnetconfig.in_channels
        hidden_channels = Resnetconfig.hidden_channels
        out_channels = Resnetconfig.out_channels
        iters = Resnetconfig.iters
        act = Resnetconfig.act
        attention = Resnetconfig.attention
        C = Resnetconfig.C
        self.conv1 = conv(in_channels=in_channels, out_channels=hidden_channels)
    
        self.convseq = nn.Sequential(*[ReconBlock(channels=hidden_channels, act=act, attention=attention, C=C) for i in range(iters)])

        self.conv3 = conv(in_channels=hidden_channels, out_channels=out_channels)

    def forward(self, x):

     
        out = self.conv1(x)
        
        out = self.convseq(out)
        out = self.conv3(out) + x
        
        return out


class simam_module(torch.nn.Module):
    """
    From https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py
    """
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with learned gaussian kernels, used to smoothen the input 
    Changed from https://github.com/BioroboticsLab/IBA/blob/master/IBA/pytorch.py 
    and https://github.com/sunkg/jCAN/blob/main/utils.py
    """

    def __init__(self):
        super().__init__()
        # self.sigma = 7
        ## Sorry for here, the parameter is not learnable if don't use any other tricks here, because it must be int number. 
        self.sigma = nn.Parameter(torch.ones(1) * 7, requires_grad=False)

    def gaussian_init(self, x):
        channels = x.shape[1]
        sigma = self.sigma
        #sigma = torch.clamp(self.sigma, min=1e-3, max=7)
        kernel_size = int(2*torch.ceil(sigma*2) + 1)
        # kernel_size = 3
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.float)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).to(x.device)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * np.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1).to(x.device)  # expand in channel dimension
        conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                            padding=0, kernel_size=kernel_size,
                            groups=channels, bias=False, device=x.device)
        conv.weight.data.copy_(kernel_3d)
        conv.weight.requires_grad = False
        pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))
        return conv, pad

    def forward(self, x):
        conv, pad = self.gaussian_init(x)
        padding = pad(x)
        out = conv(padding)
        return out 

