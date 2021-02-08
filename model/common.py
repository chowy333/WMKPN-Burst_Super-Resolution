import math

import torch
import torch.nn as nn

##############################
#    Basic layer
##############################
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

#def reflective_padding(tensor, padding):
#    B, F, N, H, W = tensor.shape
#    re_shaped = tensor.view(B, -1, H, W)
#    """Reflecting padding on H and W dimension"""
#    return nn.ReflectionPad2d((padding, padding, padding, padding))(re_shaped).view(B, F, N, H+2, W+2)

#class Ref_pad(nn.Module):
#    def __init__(self, L_pad =1, R_pad =1, T_pad= 1, B_pad = 1):
#        super(Ref_pad, self).__init__()
#
#        self.L_pad = L_pad
#        self.R_pad = R_pad
#        self.T_pad = T_pad
#        self.B_pad = B_pad
#
#        self.pad = nn.ReflectionPad2d((L_pad, R_pad, T_pad, B_pad))
#
#    def forward(self, x):
#        B, F, N, H, W = x.shape
#        reshaped = torch.reshape(x, (B, -1, H, W))
#        x = self.pad(reshaped)
#        return x.view(B, F, N, H + self.T_pad + self.B_pad, W + self.L_pad + self.R_pad)

class Zero_pad(nn.Module):
    def __init__(self, L_pad =1, R_pad =1, T_pad= 1, B_pad = 1):
        super(Zero_pad, self).__init__()

        self.L_pad = L_pad
        self.R_pad = R_pad
        self.T_pad = T_pad
        self.B_pad = B_pad

        self.pad = nn.ZeroPad2d((L_pad, R_pad, T_pad, B_pad))

    def forward(self, x):
        B, F, N, H, W = x.shape
        reshaped = torch.reshape(x, (B, -1, H, W))
        x = self.pad(reshaped)
        return x.view(B, F, N, H + self.T_pad + self.B_pad, W + self.L_pad + self.R_pad)

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
            norm_type=False, act_type='relu', is_pad = True):

        if is_pad :
            m = [Zero_pad()]
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


def default_conv(in_channelss, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(in_channelss, out_channels, kernel_size, padding=(kernel_size//2), stride=stride, bias=bias)


class ConvBlock(nn.Sequential):
    def __init__(
        self, in_channelss, out_channels, kernel_size=3, stride=1, bias=False,
            norm_type=False, act_type='relu'):

        m = [default_conv(in_channelss, out_channels, kernel_size, stride=stride, bias=bias)]
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, out_channels) if norm_type else None
        if norm:
            m.append(norm)
        if act is not None:
            m.append(act)
        super(ConvBlock, self).__init__(*m)

class ESA(nn.Module):
    def __init__(
        self, n_feats = 64, kernel_size=3,
            norm_type=False, act_type='sigmoid', bias=False, res_scale=1):

        super(ESA, self).__init__()

        self.conv1_1 = default_conv(n_feats, n_feats//4, 1, stride=1, bias=False)
        self.conv1_1_2 = default_conv(n_feats//4, n_feats , 1, stride=1, bias=False)

        m = [] # start with stride conv
        dilated_conv = nn.Conv2d(n_feats//4, n_feats//4, kernel_size, padding=0, dilation = 2, stride=1, bias=bias)
        pool = nn.MaxPool2d(7, padding = 1, stride = 3) # patch_size = 48이라서 그럼
        m.append(dilated_conv)
        m.append(pool)
        for i in range(3):
            m.append(nn.Conv2d(n_feats//4, n_feats//4, kernel_size, padding=0, stride=1, bias=bias))

        m.append(nn.Upsample(scale_factor = 6, mode = "nearest"))

        self.act = act_layer(act_type) if act_type else None
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale


    def forward(self, x):
        res  = self.conv1_1(x)
        res1 = self.body(res)
        res2 = res1 + res
        res3 = self.conv1_1_2(res2)
        attn_mask = self.act(res3)
        output = attn_mask * x
        return output

#import torch
#import torch.nn as nn
#input_shape = (1,64,48,48)
#input = torch.randn(input_shape)
#input.shape
#n_feats = 64
#kernel_size = 3
#conv1_1 = default_conv(n_feats, n_feats//4, 1, stride=1, bias=False)
#x = conv1_1(input)
#x.shape
#x = dilated_conv(x)
#x.shape
#pool = nn.MaxPool2d(7, padding = 1, stride = 3)
#x = pool(x)
#x.shape
#conv = nn.Conv2d(n_feats//4, n_feats//4, kernel_size, padding=0, stride=1)
#x.shape
#x = conv(x)
#x = conv(x)
#x = conv(x)
#x.shape
#upsample = nn.Upsample(scale_factor = 6, mode = "nearest")
#upsample(x).shape
#ESA()(input).shape

# 입력 (16, 14, 64, 48, 48) 용도
#class TCSAB(nn.Module):
#    def __init__(
#        self, n_feat = 14, kernel_size = 3, stride = 1, padding = 1, norm_type = False, #act_type = "relu", bias = True, reduction = 6):
#
#        super(TCSAB, self).__init__()
#        modules_body = []
#
#        modules_body.append(ConvBlock_3D(n_feat, n_feat, kernel_size = (3,3,3),act_type= #act_type, padding = padding, bias= bias, is_rpad = False))
#        modules_body.append(conv3d_weightnorm(n_feat, n_feat, kernel_size, stride = stride, #padding = padding, bias = bias))
#        modules_body.append(RFAB(n_feat, kernel_size, norm_type, act_type, bias, reduction))
#        modules_body.append(CSAM(n_feat))
#        self.body = nn.Sequential(*modules_body)
#
#    def forward(self, x):
#        res = self.body(x)
#        #res = self.body(x).mul(self.res_scale)
#        res += x
#        return res

# 입력 (16, 64, 14, 48, 48) 용도
class TCSAB(nn.Module):
    def __init__(
        self, n_feat = 14, kernel_size = 3, stride = 1, padding = 1, norm_type = False, act_type = "relu", bias = True, reduction = 6):

        super(TCSAB, self).__init__()
        modules_body = []

        modules_body.append(ConvBlock_3D(n_feat, n_feat, kernel_size = (3,3,3),act_type= act_type, padding = padding, bias= bias, is_pad = False))
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

class RFABlock(nn.Module):
    def __init__(
        self, n_feats = 64, kernel_size=3,
            norm_type=False, act_type='relu', bias=False, res_scale=1):

        super(RFABlock, self).__init__()
        m = []
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, n_feats) if norm_type else None

        m.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
        m.append(act)
        m.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
        m.append(ESA())

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        return res

class RFA(nn.Module):
    def __init__(
        self, n_feats = 64, kernel_size=3,
            norm_type=False, act_type='relu', bias=False, res_scale=1):

        super(RFA, self).__init__()
        self.RFAB = RFABlock()
        self.conv = default_conv(n_feats * 4, n_feats, 1, stride=1, bias=False)
    def forward(self, x):
        res1 = self.RFAB(x)
        res = res1 + x
        res2 = self.RFAB(res)
        res = res2 + res
        res3 = self.RFAB(res)
        res = res3 + res
        res4 = self.RFAB(res)
        output = self.conv(torch.cat((res1, res2, res3, res4), 1))
        output = output + x
        return output

##############################
#    Useful Blocks
##############################
class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size=3,
            norm_type=False, act_type='relu', bias=False, res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, n_feats) if norm_type else None
        for i in range(2):
            m.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm:
                m.append(norm)
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualDenseBlock5(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR18)
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(ResidualDenseBlock5, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv5 = ConvBlock(nc+4*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(self.res_scale) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.RDB1 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)
        self.RDB2 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)
        self.RDB3 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(self.res_scale) + x


class SkipUpDownBlock(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR18)
    """

    def __init__(self, nc, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(SkipUpDownBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, nc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv2 = ConvBlock(2*nc, 2*nc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.up = nn.PixelShuffle(2)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(nc, nc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.up(torch.cat((x, x1, x2), 1))
        x3 = self.conv3(self.pool(x3))
        return x3.mul(self.res_scale) + x


class DUDB(nn.Module):
    """
    Dense Up Down Block
    """

    def __init__(self, nc, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(DUDB, self).__init__()
        self.res_scale = res_scale
        self.UDB1 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm_type, act_type, res_scale)
        self.UDB2 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm_type, act_type, res_scale)
        self.UDB3 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm_type, act_type, res_scale)

    def forward(self, x):
        return self.UDB3(self.UDB2(self.UDB1(x))).mul(self.res_scale) + x


###########################
#  Upsamler layer
##########################
class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, norm_type=False, act_type='relu', bias=False):

        m = []
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, n_feats) if norm_type else None
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if norm: m.append(norm)
                if act is not None: m.append(act)

        elif scale == 3:
            m.append(default_conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if norm: m.append(norm)
            if act is not None: m.append(act)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

a = torch.randn((16, 64, 48, 48))
Up = Upsampler(4*2, 64)
Up(a).shape
class DownsamplingShuffle(nn.Module):

    def __init__(self, scale):
        super(DownsamplingShuffle, self).__init__()
        self.scale = scale

    def forward(self, input):
        """
        input should be 4D tensor N, C, H, W
        :return: N, C*scale**2,H//scale,W//scale
        """
        N, C, H, W = input.size()
        assert H % self.scale == 0, 'Please Check input and scale'
        assert W % self.scale == 0, 'Please Check input and scale'
        map_channels = self.scale ** 2
        channels = C * map_channels
        out_height = H // self.scale
        out_width = W // self.scale

        input_view = input.contiguous().view(
            N, C, out_height, self.scale, out_width, self.scale)

        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

        return shuffle_out.view(N, channels, out_height, out_width)


def demosaick_layer(input):
    demo = nn.PixelShuffle(2)
    return demo(input)


#############################
#  counting number
#
#############################
def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
