import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import argparse
#from torchsummary import summary
import torchvision.models as models
import math
import random

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

def default_conv(in_channelss, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(in_channelss, out_channels, kernel_size, padding=(kernel_size//2), stride=stride, bias=bias)


### demosaicing layer를 추가해야 될듯
class NET(nn.Module): #kpn(bst, feat)
    def __init__(self, args):
        super(NET, self).__init__()
        #self.kernel_size = [args.kernel_size]
        self.kernel_size = list(map(int, args.multi_kernel_size.split(","))) # [1, 3, 5, 7]
        self.scale = args.scale
        self.upMode = args.upMode
        self.num_seq = args.num_seq
        self.blind_est = args.blind_est
        self.sep_conv = args.sep_conv
        self.core_bias = args.core_bias
        self.num_B = args.num_B
        self.color = args.color
        channel_att = args.channel_att
        spatial_att = args.spatial_att
        self.in_channels = args.in_channels
        in_channel = (self.in_channels if self.color else 1) * (self.num_seq if self.blind_est else self.num_seq+1)
        out_channel = (2 * sum(self.kernel_size) if self.sep_conv else np.sum(np.array(self.kernel_size) ** 2)) * self.num_seq * args.in_channels * args.gt_channels
        self.rate = 1

        self.conv1 = Basic(4*14, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, 256, channel_att=channel_att, spatial_att=spatial_att)
        #self.conv9 = Basic(256+64, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        #self.out_kernel = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.out_kernel = nn.Conv2d(256+64, out_channel, 1, 1, 0)


        #self.out_weight = nn.Sequential(
        #    nn.Conv2d(128, 64, 1, 1, 0),
        #    Basic(64, 64, g=1),
        #    nn.Conv2d(64, len(self.kernel_size)*args.gt_channels, 1, 1, 0),
        #    # nn.Softmax(dim=1)  #softmax 效果较差
        #    nn.Sigmoid()
        #)

        self.out_weight = nn.Sequential(
            nn.Conv2d(256+64, 64, 1, 1, 0),
            Basic(64, 64, g=1),
            nn.Conv2d(64, self.num_seq * len(self.kernel_size) * args.gt_channels, 1, 1, 0),
        )
        #(B, 12, 48, 48)
        #(B, 4, 3, 48, 48)
        self.softmax = nn.Softmax(dim=2)

        self.kernel_pred = KernelConv(self.kernel_size, self.sep_conv, self.core_bias)
        self.SR_recon = SR_Rec(in_channel = 3*14, up_scale= 8)
        #self.SR_recon = SR_Rec(in_channel = 3*14*4)

        res = []
        res.append(nn.Conv2d(4*14, 64, 3, 1, 1))
        res.append(Upsampler(8, 64))
        #res.append(nn.Conv2d(64, 3, 3, padding=1, bias=True))
        self.res = nn.Sequential(*res)
        self.last_conv = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        B, N, C, H, W = data.shape
        #print("asdlf ", data_with_est.shape)
        conv1 = self.conv1(data_with_est)
        #print("conv1 =", conv1.shape) # (B, 64, 48, 48)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        #print("conv2 =", conv2.shape) # (B, 128, 24, 24)
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        #print("conv3 =", conv3.shape) # (B, 256, 12, 12)
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        #print("conv4 =", conv4.shape) # (B, 512, 6, 6)
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        #print("conv5 =", conv5.shape) # (B, 512, 3, 3)
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        #print("conv6 =", conv6.shape) # (B, 512, 6, 6)
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        #print("conv7 =", conv7.shape) # (B, 256, 12, 12)
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        #print("conv8 =", conv8.shape) # (B, 256, 24, 24)
        core = self.out_kernel(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        #print("conv9 =", conv9.shape) # (B, 5376, 48, 48)
        #core = self.out_kernel(conv9)
        #print("core", core.shape) # (B, 5376, 48, 48)
        kernel_weight = self.out_weight(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        #print("aslkjdfasd", kernel_weight.shape) # [2, 168, 48, 48]
        kernel_weight = self.softmax(kernel_weight.view(B, N, len(self.kernel_size), 3, H, W)) #(B, 14, 4, 3, 48, 48)
        #print("qlwkjdf", kernel_weight.shape) # [2, 14, 4, 3, 48, 48]
        pred_i, pred = self.kernel_pred(data, core, kernel_weight, white_level, self.rate)

        res = self.res(data_with_est)
        #print("aslkdjfasd",pred_i.view(B, -1, H, W).shape)
        SR_out = self.SR_recon(pred_i.view(B, -1, H, W)) + res
        return pred_i, pred, self.last_conv(SR_out)

class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm

class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, 3, height, width)

        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, color, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, color, 3, height, width)

            core_out[K] = torch.einsum('ijklcnop,ijlmcnop->ijkmcnop', [t1, t2]).view(batch_size, N, K, K, color, 3, height, width)

            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, 3, 4, height, width) # (16, 14 ,5^2, 1, 48, 48)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...] # (16, 14 ,1, 48, 48) Flase 면 bias = None
        return core_out, bias # core_out[5] = (16, 14 ,5^2, 3, 48, 48), bias = None

    def forward(self, frames, core, kernel_weight, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)

        img_stack = []
        pred_rgb = []
        kernel = self.kernel_size[::-1] # [7, 5, 3, 1]
        padded_kernel = [core[kernel[0]]]
        iterval = iter(kernel)
        p_kernel = next(iterval)

        for index, p_kernel in enumerate(iterval): #5, 3, 1

            padding_num = (kernel[0] - p_kernel) // 2
            padded = F.pad(core[p_kernel], [0, 0, 0, 0, 0, 0, 0, 0, padding_num, padding_num, padding_num, padding_num])
            padded_kernel.append(padded)

        padded_kernel = torch.stack(padded_kernel, dim=2) # (B, 14, 4, 7, 7, 4, 3, 48, 48) #kernel_weight : #(B, 14, 4, 1, 3, 48, 48)
        #print(padded_kernel[..., 0, :, :].view(batch_size, N, len(kernel), -1, height, width).shape) #(B, 14, 4, 7*7*4, 3, 48, 48)
        #print("wlkjsf", padded_kernel.view(batch_size, N, 4, -1, 3, 48, 48).shape, kernel_weight.shape) # (B, 14, 4, 7*7*4, 3, 48, 48) #kernel_weight : #(B, 14, 4, 3, 48, 48)

        weighted_kernel = torch.mean(padded_kernel.view(batch_size, N, 4, -1, 3, 48, 48).mul(kernel_weight.unsqueeze(dim=3)), dim=2) # (B, 14, 4, 7*7*4, 3, 48, 48)
        #print(weighted_kernel.shape) torch.Size([B, 14, 196, 3, 48, 48])

        if len(img_stack) == 0:
            frame_pad = F.pad(frames, [kernel[0] // 2, kernel[0] // 2, kernel[0] // 2, kernel[0] // 2])
            for i in range(kernel[0]):
                for j in range(kernel[0]):
                    img_stack.append(frame_pad[..., i:i + height, j:j + width])
            img_stack = torch.stack(img_stack, dim=2)
        else:
            k_diff = (kernel[index - 1]**2 - kernel[index]**2) // 2
            img_stack = img_stack[:, :, k_diff:-k_diff, ...]
        #img_stack (B, 14, 49, 4, 48, 48)
        for i in range(3): # RGB(B, 14, 4, )
            pred_rgb.append(torch.sum(torch.sum(weighted_kernel[..., i, :, :].view(batch_size, N, -1, len(kernel), height, width).mul(img_stack), dim=2, keepdim = False), dim =2, keepdim = False))

        pred_img_i = torch.stack(pred_rgb, dim=2) # (B, 14, 3, 48, 48)
        pred_img = torch.mean(pred_img_i, dim =1, keepdim=False)

        return pred_img_i, pred_img # yi(16 ,14, 3, 48, 48), y(16 ,3, 48, 48)

class Res_Block_s(nn.Module):
    def __init__(self, scale=1.0):
        super(Res_Block_s, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.scale = scale

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res.mul(self.scale)

class SR_Rec(nn.Module):
    def __init__(self, in_channel, up_scale = 8, nb_block=10, scale=1.0):
        super(SR_Rec, self).__init__()
        self.recon_layer = self.make_layer(Res_Block_s(scale), nb_block)
        fea_ex = [nn.Conv2d(in_channel, 64, 3, padding=1, bias=True),
                  nn.ReLU()]
        self.fea_ex = nn.Sequential(*fea_ex)
        upscaling = [
            Upsampler(up_scale, 64, act_type="relu")]
        self.up = nn.Sequential(*upscaling)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, y):
        batch_size, ch, w, h = y.size()
        #center = num // 2
        #y_center = y[:, center, :, :, :]
        y = y.view(batch_size, -1, w, h)
        fea = self.fea_ex(y)
        out = self.recon_layer(fea)
        out = self.up(out)
        return out #+ F.upsample(y_center, scale_factor=4, mode='bilinear')

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, norm_type=False, act_type='relu', bias=True):

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

class LossFunc(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)


class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        #print(pred.shape, ground_truth.shape)
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))


class LossAnneal(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #core[5] = torch.randn((4, 3*64*48*48, 56*25)).to(device)
    #stack = torch.randn((4, 48*48, 56*25)).to(device)
    #pred = torch.matmul(core[5], stack).shape
    #print(pred)
    parser = argparse.ArgumentParser(description='PyTorch implementation of KPN-BurstSR')
    parsers = BaseArgs()
    args = parsers.initialize(parser)

    bs = torch.randn((8, 14*4, 48, 48)).to(device)
    feat = torch.randn((8, 14, 4, 48, 48)).to(device)
    print("start")
    kpn = NET(args).to(device)
    #print("start")
    print(kpn(bs, feat)[2].shape)

    #kpn = KPN(6, 5*5*6, True, True).cuda()
    #print(summary(kpn, (6, 224, 224), batch_size=4))
