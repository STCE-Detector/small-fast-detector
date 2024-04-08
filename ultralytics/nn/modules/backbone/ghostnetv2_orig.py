# 2020.11.06-Changed for building GhostNetV2
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Creates a GhostNet Model as defined in:
# GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
# https://arxiv.org/abs/1911.11907
# Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models

# Taken from https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
# The file is modified by Deeplite Inc. from the original implementation on Jan 18, 2023
# Code implementation refactoring

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Mish

__all__ = (
    'GhostModuleV2_orig',
    'GhostBottleneckV2_orig',
    'GhostConv2_orig',
    'ConvBnAct',
    'SELayer',
    'get_activation',
    'round_channels',
    'autopad',
)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

ACT_TYPE_MAP = {
    'relu': nn.ReLU(inplace=True),
    'relu6': nn.ReLU6(inplace=True),
    'hswish': nn.Hardswish(inplace=True),
    'hardswish': nn.Hardswish(inplace=True),
    'silu': nn.SiLU(inplace=True),
    'lrelu': nn.LeakyReLU(0.1, inplace=True),
    'hsigmoid': nn.Hardsigmoid(inplace=True),
    'sigmoid': nn.Sigmoid(),
    'mish': Mish(),
    'leakyrelu': nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'leakyrelu_0.1': nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'gelu': nn.GELU(),
}


def get_activation(activation_name):
    if activation_name:
        return ACT_TYPE_MAP[activation_name]
    return nn.Identity()


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def round_channels(channels, divisor=8):
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    mid_channels : int or None, default None
        Number of middle channels.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    activation : function, or str, or nn.Module, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or nn.Module, default 'sigmoid'
        Activation function after the last convolution.
    """

    def __init__(
            self,
            channels,
            reduction=16,
            mid_channels=None,
            round_mid=False,
            use_conv=True,
            mid_activation='relu',
            out_activation='hsigmoid',
            norm_layer=None,
    ):
        super(SELayer, self).__init__()
        print("SELayer")
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = (
                channels // reduction
                if not round_mid
                else round_channels(float(channels) / reduction, round_mid)
            )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=True,
            )
        else:
            self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)

        self.bn = norm_layer(mid_channels) if norm_layer else nn.Identity()
        self.activ = get_activation(mid_activation)

        if use_conv:
            self.conv2 = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=True,
            )
        else:
            self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)
        self.sigmoid = get_activation(out_activation)

    def forward(self, x):
        w = self.pool(x)

        # timm:
        # w = x.mean((2, 3), keepdim=True)
        # if self.add_maxpool:
        # # experimental codepath, may remove or change
        #   w = 0.5 * w + 0.5 * x.amax((2, 3), keepdim=True)

        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(self.bn(w))
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x


class ConvBnAct(nn.Module):
    # Standard convolution-batchnorm-activation block
    def __init__(
            self,
            c1,  # input channels
            c2,  # output channels
            k=1,  # kernel size
            s=1,  # stride
            p=None,  # padding
            g=1,  # groups
            b=None,  # bias
            act='relu',  # activation, either a string or a nn.Module; nn.Identity if None
            d=1,  # dilation
            residual=False,  # whether do add a skip connection
            use_bn=True,  # whether to use BatchNorm
            channel_divisor=1,  # round the number of out channels to the nearest multiple of channel_divisor
    ):
        super().__init__()

        # YOLOv5 applies channel_divisor=8 by default
        c2 = round_channels(c2, channel_divisor)

        self.in_channels = c1
        self.out_channels = c2
        self.use_bn = use_bn
        b = not self.use_bn if b is None else b

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=b)

        self.bn = nn.BatchNorm2d(c2) if use_bn else nn.Identity()
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()
        self.residual = residual

        self.resize_identity = (c1 != c2) or (s != 1)

        if self.residual:
            # in case the input and output shapes are different, we need a 1x1 conv in the skip connection
            self.identity_conv = nn.Sequential()
            self.identity_conv.add_module(
                'conv', nn.Conv2d(c1, c2, 1, s, autopad(1, p), bias=b)
            )
            if self.use_bn:
                self.identity_conv.add_module('bn', nn.BatchNorm2d(c2))

    def forward(self, x):
        inp = x
        out = self.act(self.bn(self.conv(x)))
        if self.residual:
            if self.resize_identity:
                out = out + self.identity_conv(inp)
            else:
                out = out + inp
        return out

    def forward_fuse(self, x):
        inp = x
        out = self.act(self.conv(x))
        if self.residual:
            if self.resize_identity:
                out = out + self.identity_conv(inp)
            else:
                out = out + inp
        return out


class DFCModule(nn.Module):
    def __init__(self, c1, c2, k, s, dfc_k=5, downscale=False):
        super().__init__()
        self.downscale = downscale
        self.gate_fn = nn.Sigmoid()
        self.short_conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(
                c2,
                c2,
                kernel_size=(1, dfc_k),
                stride=1,
                padding=(0, 2),
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.Conv2d(
                c2,
                c2,
                kernel_size=(dfc_k, 1),
                stride=1,
                padding=(2, 0),
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        res = F.avg_pool2d(x, kernel_size=2, stride=2) if self.downscale else x
        res = self.short_conv(res)
        res = self.gate_fn(res)
        if self.downscale:
            res = F.interpolate(res, size=(x.shape[-2], x.shape[-1]), mode='nearest')
        return res


class GhostModuleV2_orig(nn.Module):
    def __init__(self, c1, c2, k=1, ratio=2, dw_k=3, s=1, dfc=True, act='relu', downscale=False):
        super(GhostModuleV2_orig, self).__init__()
        self.dfc = dfc
        self.act = get_activation(act)
        self.downscale = downscale
        self.oup = c2
        init_channels = math.ceil(c2 / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, init_channels, k, s, k // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            self.act,
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_k,
                1,
                dw_k // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            self.act,
        )

        if self.dfc:
            self.dfc = DFCModule(c1, c2, k, s, downscale=self.downscale)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        res = out[:, :self.oup, :, :]
        if self.dfc:
            res = res * self.dfc(x)
        return res


class GhostBottleneckV2_orig(nn.Module):
    def __init__(
            self,
            c1,
            c2,
            dw_kernel_size=3,
            s=1,
            se_ratio=0,
            layer_id=None,
            act='relu',
            downscale=False
    ):
        super(GhostBottleneckV2_orig, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = s
        self.act = get_activation(act)
        c_ = c2 // 2
        self.downscale = downscale
        print("Downscale Bottleneck", self.downscale)

        # point-wise expansion
        do_dfc = True if layer_id is None else layer_id > 1
        self.ghost1 = GhostModuleV2_orig(c1, c_, dfc=do_dfc, act=act, downscale=self.downscale)

        # depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                c_,
                c_,
                dw_kernel_size,
                stride=s,
                padding=(dw_kernel_size - 1) // 2,
                groups=c_,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(c_)

        # squeeze-and-excitation
        self.se = (
            SELayer(c_, reduction=int(1 / se_ratio), round_mid=4)
            if has_se
            else None
        )

        self.ghost2 = GhostModuleV2_orig(c_, c2, dfc=False, act=None, downscale=False)

        # shortcut
        if c1 == c2 and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    c1,
                    c1,
                    dw_kernel_size,
                    stride=s,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=c1,
                    bias=False,
                ),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c2, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostConv2_orig(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
            self,
            c1,
            c2,
            k=1,
            s=1,
            g=1,
            dw_k=5,
            dw_s=1,
            act='relu',
            shrink_factor=0.5,
            residual=False,
            dfc=False,
            downscale=False,
    ):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv2_orig, self).__init__()
        c_ = int(c2 * shrink_factor)  # hidden channels

        self.residual = residual
        self.dfc = None
        self.downscale = downscale
        print("Downscale DFC MODULE", self.downscale)
        if dfc:
            self.dfc = DFCModule(c1, c2, k, s, downscale=self.downscale)

        self.single_conv = False
        if c_ < 2:
            self.single_conv = True
            self.cv1 = ConvBnAct(c1, c2, k, s, p=None, g=g, act=act)
        else:
            self.cv1 = ConvBnAct(c1, c_, k, s, p=None, g=g, act=act)
            self.cv2 = ConvBnAct(c_, c_, dw_k, dw_s, p=None, g=c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        if self.single_conv:
            return y
        res = torch.cat([y, self.cv2(y)], 1) if not self.residual \
            else x + torch.cat([y, self.cv2(y)], 1)
        if self.dfc:
            res = res * self.dfc(x)
        return res