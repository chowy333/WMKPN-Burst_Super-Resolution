import numpy as np
import torch
import torch.nn as nn
import os
from model.common import ConvBlock_2D, ConvBlock_3D, Ref_pad, RFAB, RTAB, Zero_pad
#from common import ConvBlock_2D, ConvBlock_3D, Ref_pad, RFAB, RTAB

class NET(nn.Module):
    def __init__(self, args):
        super(NET, self).__init__()
        filters = args.filters
        num_rfab = args.num_rfab
        scale = args.scale
        num_seq = args.num_seq
        denoise = args.denoise
        #block_type = args.block_type
        act_type = args.act_type
        bias = args.bias
        norm_type = args.norm_type
        r_rate = args.reduction
        in_channels = args.in_channels

        # define sr module noise map
        if denoise:
            m_sr_head = [ConvBlock_3D(5, filters, 3, act_type=None, bias=bias, is_rpad = True)]
        else:
            m_sr_head = [ConvBlock_3D(4, filters, 3, act_type=None, bias= bias, is_rpad = True)]

        m_rfab = []
        for _ in range(num_rfab):
            m_rfab += [RFAB(n_feats = filters, kernel_size=3, act_type = act_type, bias = bias, reduction = r_rate)]
            #m_sr_resblock += [Self_Attn(sr_n_feats)]

        m_rfab += [ConvBlock_3D(filters, filters, 3, act_type=None, bias=bias, is_rpad = False)]

        m_trb = []
        for i in range(0, np.floor_divide(num_seq,2)-1):# 2씩 줄어드는거 6번
        #for i in range(0, np.floor_divide(6)):
            m_trb += [Zero_pad()]
            m_trb += [RFAB(n_feats = filters, kernel_size=3, act_type = act_type, bias = bias, reduction = r_rate)]
            m_trb += [ConvBlock_3D(filters, filters, kernel_size = (3,3,3), act_type= act_type, padding = 0, bias= bias, is_rpad = False)]


        #m_sr_up = [Upsampler(scale, filters, norm_type, act_type, bias=bias), ConvBlock(filters, 3, 3, bias=True)]

        m_sr_tail = [ConvBlock_2D(filters*2, (scale**2)*filters, (3,3), act_type=None, padding = 0, bias=bias), nn.PixelShuffle(scale*2)]


        self.head = nn.Sequential(*m_sr_head)

        self.trunk_rfab = nn.Sequential(*m_rfab)
        self.trunk_trb = nn.Sequential(*m_trb)
        self.tail = nn.Sequential(*m_sr_tail)
        self.tail_conv = ConvBlock_2D(filters//4, 3, (3,3), act_type=None, padding = 1, bias=bias)

        m_global_res = [nn.ZeroPad2d((1,1,1,1)), RTAB(num_seq * in_channels), ConvBlock_2D(num_seq * in_channels, (scale**2)*filters, kernel_size = (3,3), padding =0, bias = bias), nn.PixelShuffle(scale*2)]
        self.global_res = nn.Sequential(*m_global_res)


        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_normal_(m.weight)
        #        m.weight.requires_grad = True
        #        if m.bias is not None:
        #            m.bias.data.zero_()
        #            m.bias.requires_grad = True


    def forward(self, x):
        B, F, T, H, W = x.shape
        #print("x", x.shape)
        global_res = self.global_res(torch.reshape(x, (B, -1, H, W)))
        x = self.head(x)
        #print("after head", x.shape)
        x_res = x
        x = self.trunk_rfab(x)
        #print("after rfab", x.shape)
        x = x + x_res
        x = self.trunk_trb(x)
        #print("after trb", x.shape)
        x = x.view(B, -1, H+2, W+2)
        sr_output = self.tail(x)
        #print("Final", sr_output.shape, global_res.shape)
        return self.tail_conv(sr_output + global_res)


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
