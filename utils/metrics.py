import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.spatial_color_alignment as sca_utils
from utils.warp import warp
import cv2
import skimage.measure as measure
class L2(nn.Module):
    def __init__(self, boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_m = pred
        gt_m = gt

        if valid is None:
            mse = F.mse_loss(pred_m, gt_m)
        else:
            mse = F.mse_loss(pred_m, gt_m, reduction='none')

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse


class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = L2(boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, valid=None):
        assert pred.dim() == 4 and pred.shape == gt.shape
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr


class AlignedL2(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear') * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear')

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = sca_utils.match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        mse = F.mse_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse


class AlignedPSNR(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, burst_input):
        mse = self.l2(pred, gt, burst_input)

        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, burst_input):
        psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr

def calc_metrics(img1, img2, crop_border, test_Y=True): # 0 ~ 1


    if test_Y and img1.shape[1] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return ssim

def torch2numpy(tensor, gamma=None):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    tensor *= 255.0
    while len(tensor.size()) < 4:
        tensor = tensor.unsqueeze(1)
    return tensor.permute(0, 2, 3, 1).cpu().data.numpy()

def calculate_ssim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    ssim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim += measure.compare_ssim(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...],
                                             multichannel=True,
                                             data_range=255)
        n += 1.0
    return ssim / n

#out = torch.randn((16,4,48,48))
#tar = torch.randn((16,4,48,48))
#calculate_ssim(out, tar)
def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
