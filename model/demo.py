import numpy as np
import torch
import torch.nn as nn
import os
B = 16
N = 8 # num_block
filterss = 64
kernel_size = 3
num_seqs = 14 # temporal
r = 8
scaled = 4

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randn((16, 4, 14 ,48, 48)).to(device)

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

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
            m_sr_head = [ConvBlock_3D(4, filters, 3, act_type=None, bias= bias, is_rpad = True)]

        m_rfab = []
        for _ in range(num_rfab):
            m_rfab += [RFAB(n_feats = filters, kernel_size=3, act_type = act_type, bias = bias, reduction = r_rate)]
            #m_sr_resblock += [Self_Attn(sr_n_feats)]

        m_rfab += [ConvBlock_3D(filters, filters, 3, act_type=None, bias=bias, is_rpad = False)]

        m_trb = []
        for i in range(0, np.floor_divide(num_seq,2)-1):# 2씩 줄어드는거 6번
        #for i in range(0, np.floor_divide(6)):
            m_trb += [Ref_pad()]
            m_trb += [RFAB(n_feats = filters, kernel_size=3, act_type = act_type, bias = bias, reduction = r_rate)]
            m_trb += [ConvBlock_3D(filters, filters, kernel_size = (3,3,3), act_type= act_type, padding = 0, bias= bias, is_rpad = False)]

        m_sr_tail = [ConvBlock_2D(filters*2, (scale*2)**2*3, (3,3), act_type=None, padding = 0, bias=bias), nn.PixelShuffle(scale*2)]


        self.head = nn.Sequential(*m_sr_head)

        self.trunk_rfab = nn.Sequential(*m_rfab)
        self.trunk_trb = nn.Sequential(*m_trb)
        self.tail = nn.Sequential(*m_sr_tail)


        m_global_res = [nn.ReflectionPad2d((1,1,1,1)), RTAB(56), ConvBlock_2D(56, (scale*2)**2*3, kernel_size = (3,3), padding =0, bias = bias), nn.PixelShuffle(scale*2)]
        self.global_res = nn.Sequential(*m_global_res)


    def forward(self, x):
        B, F, T, H, W = x.shape
        print("x", x.shape)
        global_res = self.global_res(x.view(B, -1, H, W))
        x = self.head(x)
        print("after head", x.shape)
        x_res = x
        x = self.trunk_rfab(x)
        print("after rfab", x.shape)
        x = x + x_res
        x = self.trunk_trb(x)
        print("after trb", x.shape)
        x = x.view(B, -1, H+2, W+2)
        sr_output = self.tail(x)
        print("Final", sr_output.shape)
        return sr_output + global_res

4**2*3
model = NET().to(device)
model(input).shape
k = torch.randn((16,192, 48, 48))
ps = nn.PixelShuffle(8)
ps(k).shape
conv = ConvBlock_2D(128, 3, kernel_size = 3, padding = 0)
conv(k).shape
del(input)
del(model)
torch.cuda.empty_cache()

class Ref_pad(nn.Module):
    def __init__(self, L_pad =1, R_pad =1, T_pad= 1, B_pad = 1):
        super(Ref_pad, self).__init__()

        self.L_pad = L_pad
        self.R_pad = R_pad
        self.T_pad = T_pad
        self.B_pad = B_pad

        self.pad = nn.ReflectionPad2d((L_pad, R_pad, T_pad, B_pad))

    def forward(self, x):
        B, F, N, H, W = x.shape
        reshaped = x.view(B, -1, H, W)
        x = self.pad(reshaped)
        return x.view(B, F, N, H + self.T_pad + self.B_pad, W + self.L_pad + self.R_pad)

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

class ConvBlock_2D(nn.Sequential):
    def __init__(
        self, in_feat, out_feat, kernel_size=3, stride=1, padding = 1, bias=False,
            norm_type=False, act_type='relu'):

        m = [conv2d_weightnorm(in_feat, out_feat, kernel_size, stride=stride, padding = padding, bias=bias)]
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, out_channels) if norm_type else None
        if norm:
            m.append(norm)
        if act is not None:
            m.append(act)
        super(ConvBlock_2D, self).__init__(*m)

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

class RTAB(nn.Module):
    def __init__(self, n_feats = 14, kernel_size=3, norm_type=False, act_type='relu', bias= False, reduction= 8): # 9는 burst frame 개수

        super(RTAB, self).__init__()

        act = act_layer(act_type) if act_type else None

        m = []
        m.append(conv2d_weightnorm(n_feats, n_feats, kernel_size, bias=bias))
        m.append(act)
        m.append(conv2d_weightnorm(n_feats, n_feats, kernel_size, bias=bias))

        n = []
        n.append(conv2d_weightnorm(n_feats, n_feats//reduction, 1, padding = 0, bias=bias))
        n.append(act)
        n.append(conv2d_weightnorm(n_feats//reduction, n_feats, 1, padding = 0,  bias=bias))
        n.append(act_layer("sigmoid"))

        self.conv_b1 = nn.Sequential(*m)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_b2 = nn.Sequential(*n)

    def forward(self, x):

        x_res = x
        x = self.conv_b1(x)
        x_to_scale = x
        x = self.avg_pool(x)
        x = self.conv_b2(x)

        x_scaled = x_to_scale * x
        return x_scaled + x_res
