import numpy as np
import torch
import torch.nn as nn
import os
from common import ConvBlock_2D, ConvBlock_3D, Ref_pad, RFAB, RTAB

B = 16
N = 8 # num_block
filterss = 32
kernel_size = 3
num_seqs = 9 # temporal
r = 8
scaled = 4

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randn((16, 1, 9 ,48, 48)).to(device)
input = torch.randn((16, 1, 9 ,48, 48))
input.permute(0,2,1,3,4).shape
class RAMS(nn.Module):
    def __init__(self):
        super(RAMS, self).__init__()

        num_rfab = N
        filters = filterss
        scale = scaled
        num_seq = num_seqs
        denoise = False
        #block_type = args.block_type
        act_type = "relu"
        bias = False
        norm_type = "None"
        r_rate = 8
        # define sr module noise map
        if denoise:
            m_sr_head = [ConvBlock_3D(5, filters, 3, act_type=None, bias=bias, is_rpad = True)]
        else:
            m_sr_head = [ConvBlock_3D(1, filters, 3, act_type=None, bias= bias, is_rpad = True)]

        m_rfab = []
        for _ in range(num_rfab):
            m_rfab += [RFAB(n_feats = filters, kernel_size=3, act_type = act_type, bias = bias, reduction = r_rate)]
            #m_sr_resblock += [Self_Attn(sr_n_feats)]

        m_rfab += [ConvBlock_3D(filters, filters, 3, act_type=None, bias=bias, is_rpad = False)]

        m_trb = []
        for i in range(0, np.floor_divide(num_seq,3)):# 2씩 줄어드는거 6번
        #for i in range(0, np.floor_divide(6)):
            m_trb += [Ref_pad()]
            m_trb += [RFAB(n_feats = filters, kernel_size=3, act_type = act_type, bias = bias, reduction = r_rate)]
            m_trb += [ConvBlock_3D(filters, filters, kernel_size = (3,3,3), act_type= act_type, padding = 0, bias= bias, is_rpad = False)]


        #m_sr_up = [Upsampler(scale, filters, norm_type, act_type, bias=bias), ConvBlock(filters, 3, 3, bias=True)]

        m_sr_tail_conv = [ConvBlock_3D(filters, scale**2, (3,3,3), act_type=None, padding = 0, bias=bias, is_rpad = False)]
        # branch for sr_raw output
        m_sr_tail_ps = [nn.PixelShuffle(4)]

        self.head = nn.Sequential(*m_sr_head)

        self.trunk_rfab = nn.Sequential(*m_rfab)
        self.trunk_trb = nn.Sequential(*m_trb)
        self.tail_conv = nn.Sequential(*m_sr_tail_conv)
        self.tail_ps = nn.Sequential(*m_sr_tail_ps)

        m_global_res = [Ref_pad(), RTAB(num_seq), ConvBlock_2D(num_seq, scale**2, kernel_size = (3,3), padding =0, bias = bias), nn.PixelShuffle(scale)]
        self.global_res = nn.Sequential(*m_global_res)

    def forward(self, x):
        x = self.head(x)
        x_res = x
        x = self.trunk_rfab(x)
        x = x + x_res
        x = self.trunk_trb(x)
        x = self.tail_conv(x)
        sr_output = self.tail_ps(torch.squeeze(x, 2))
        return sr_output



def conv3d_weightnorm(in_filters, out_filters, kernel_size, stride = 1, padding = 1, bias = False):
    return nn.utils.weight_norm(nn.Conv3d(in_filters, out_filters, kernel_size, stride = stride, padding=padding, bias = bias))

def conv2d_weightnorm(in_filters, out_filters, kernel_size, stride = 1, padding = 1, bias = False):
    return nn.utils.weight_norm(nn.Conv2d(in_filters, out_filters, kernel_size, stride = stride, padding=padding))

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

model = RAMS().to(device)
model(input).shape

#del(input)
#del(model)
#torch.cuda.empty_cache()
