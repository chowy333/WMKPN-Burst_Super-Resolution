import numpy as np
import torch
import torch.nn as nn
import os

class CSAM(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM, self).__init__()

        self.conv = nn.Conv3d(in_dim, in_dim, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.sigmoid(self.conv(x))
        out = self.gamma*out
        x = x * out + x
        return x

class RFAB(nn.Module):
    def __init__(self, n_feats = 32, kernel_size=3, norm_type=False, act_type='relu', bias= False, reduction= 8):

        super(RFAB, self).__init__()

        act = act_layer(act_type) if act_type else None

        m = []
        m.append(conv3d_weightnorm(n_feats, n_feats, kernel_size, bias=bias))
        m.append(act)
        m.append(conv3d_weightnorm(n_feats, n_feats, kernel_size, bias=bias))

        n = []
        n.append(conv3d_weightnorm(n_feats, n_feats//reduction, 1, padding = 0, bias=bias))
        n.append(act)
        n.append(conv3d_weightnorm(n_feats//reduction, n_feats, 1, padding = 0,  bias=bias))
        n.append(act_layer("sigmoid"))

        self.conv_b1 = nn.Sequential(*m)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_b2 = nn.Sequential(*n)

    def forward(self, x):
        x_res = x
        x = self.conv_b1(x)
        x_to_scale = x
        x = self.avg_pool(x)
        x = self.conv_b2(x)
        x_scaled = x_to_scale * x
        return x_scaled + x_res

class ConvBlock_3D(nn.Sequential):
    def __init__(
        self, in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False,
            norm_type=False, act_type='relu', is_rpad = True):

        if is_rpad :
            m = [Ref_pad()]
        else:
            m = []
        m += [conv3d_weightnorm(in_feat, out_feat, kernel_size, stride = stride, padding = padding, bias = bias)]
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, out_channels) if norm_type else None
        if norm:
            m.append(norm)
        if act is not None:
            m.append(act)
        super(ConvBlock_3D, self).__init__(*m)

class TCSAB(nn.Module):
    def __init__(
        self, n_feat = 14, kernel_size = 3, stride = 1, padding = 1, norm_type = False, act_type = "relu", bias = True, reduction = 6):

        super(TCSAB, self).__init__()
        modules_body = []

        modules_body.append(ConvBlock_3D(n_feat, n_feat, kernel_size = (3,3,3),act_type= act_type, padding = padding, bias= bias, is_rpad = False))
        modules_body.append(conv3d_weightnorm(n_feat, n_feat, kernel_size, stride = stride, padding = padding, bias = bias))
        modules_body.append(RFAB(n_feat, kernel_size, norm_type, act_type, bias, reduction))
        modules_body.append(CSAM(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        # 16, 64, 14, 48, 48
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # 16, 14, 64, 48, 48
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res.permute(0, 2, 1, 3, 4).contiguous()

def conv3d_weightnorm(in_filters, out_filters, kernel_size, stride = 1, padding = 1, bias = False):
    return nn.utils.weight_norm(nn.Conv3d(in_filters, out_filters, kernel_size, stride = stride, padding=padding, bias = bias))

def conv2d_weightnorm(in_filters, out_filters, kernel_size, stride = 1, padding = 1, bias = False):
    return nn.utils.weight_norm(nn.Conv2d(in_filters, out_filters, kernel_size, stride = stride, padding=padding, bias = bias))

def act_layer(act_type, inplace= True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm_layer(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer

input = torch.randn((16, 64, 14, 48, 48))
tc = TCSAB()
tc(input).shape



input.permute(0, 2, 1, 3, 4).contiguous().shape
