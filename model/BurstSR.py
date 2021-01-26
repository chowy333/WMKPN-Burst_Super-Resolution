import math
import torch
import torch.nn as nn

class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()

        sr_n_resblocks = opt.sr_n_resblocks
        sr_n_feats = opt.channels
        scale = opt.scale

        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type

        # define sr module
        if denoise:
            m_sr_head = [ConvBlock(5, sr_n_feats, 5,
                                          act_type=act_type, bias=True)]
        else:
            m_sr_head = [ConvBlock(4, sr_n_feats, 5,
                                          act_type=act_type, bias=True)]

        m_sr_resblock = []
        for _ in range(sr_n_resblocks):
            m_sr_resblock += [RRDB(sr_n_feats, sr_n_feats, 3,
                                     1, bias, norm_type, act_type, 0.2)]
            m_sr_resblock += [Self_Attn(sr_n_feats)]



        m_sr_resblock += [ConvBlock(sr_n_feats, sr_n_feats, 3, bias=bias)]

        m_sr_up = [Upsampler(scale, sr_n_feats, norm_type, act_type, bias=bias),
                   ConvBlock(sr_n_feats, 4, 3, bias=True)]

        # branch for sr_raw output
        m_sr_tail = [nn.PixelShuffle(2)]


        self.model_sr = nn.Sequential(*m_sr_head,
                                      ShortcutBlock(nn.Sequential(*m_sr_resblock)),
                                      *m_sr_up)
        self.sr_output = nn.Sequential(*m_sr_tail)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model_sr(x)
        sr_raw = self.sr_output(x)
        return sr_raw

    def get_attention(self, x):
        out = self.model_sr[0](x)
        attention_maps = []
        out_sub = self.model_sr[1].sub[0](out)
        for ii in range(1, 13):
            if ii % 2 == 0:
                out_sub = self.model_sr[1].sub[ii](out_sub)
            else:
                out_sub, attn = self.model_sr[1].sub[ii](out_sub, get_attn_map=True)
                attention_maps.append(attn.cpu().detach().numpy())
        out = self.model_sr[2](out_sub + out)
        out = self.model_sr[3](out)
        out = self.sr_output(out)
        return out, attention_maps

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        """
        in_dim   -- input feature's channel dim
        activation    -- activation function type
        """
        super(Self_Attn, self).__init__()
        ## Your Implementation Here ##
        self.k = 8    # reduce channel scale
        self.f_conv = nn.Conv2d(in_dim , in_dim//self.k, 1)
        self.g_conv = nn.Conv2d(in_dim , in_dim//self.k, 1)
        self.h_conv = nn.Conv2d(in_dim , in_dim//self.k, 1)
        self.v_conv = nn.Conv2d(in_dim//self.k, in_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, get_attn_map=False):
        ## Your Implementation Here ##
        B, C, H, W = x.shape
        N = H * W

        f_x = self.f_conv(x) # (B, C/k, H, W)
        g_x = self.g_conv(x) # (B, C/k, H, W)
        h_x = self.h_conv(x) # (B, C/k, H, W)

        # Equation (1)
        attn_map = self.softmax(torch.bmm(                    # permute -> transpose per each batch
                        f_x.view(B, -1, N).permute(0, 2, 1),  # (B, N, C/k)
                        g_x.view(B, -1, N)                    # (B, C/k, N)
                        ))                                    # (B, N, N)

        # Equation (2)
        o = self.v_conv(torch.bmm(
                        h_x.view(B, -1, N),                   # (B, C/k, N)
                        attn_map.permute(0, 2, 1)             # (B, N, N)
                        ).view(B, -1, H, W))                  # (B, C, H, W)

        # Equation (3)
        y = self.gamma*o + x

        if get_attn_map:
            return y, attn_map
        else:
            return y

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

##############################
#    Basic layer
##############################
def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
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


def default_conv(in_channelss, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size//2), stride=stride, bias=bias)


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

#############################
#  counting number
#############################
def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
