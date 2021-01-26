import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
# from model.common import DownsamplingShuffle
import pdb


def _sigmoid_to_tanh(x):
    """
    range [0, 1] to range [-1, 1]
    :param x: tensor type
    :return: tensor
    """
    return (x - 0.5) * 2.


def _tanh_to_sigmoid(x):
    """
    range [-1, 1] to range [0, 1]
    :param x:
    :return:
    """
    return x * 0.5 + 0.5


def _255_to_tanh(x):
    """
    range [0, 255] to range [-1, 1]
    :param x:
    :return:
    """
    return (x - 127.5) / 127.5


def _tanh_to_255(x):
    """
    range [-1. 1] to range [0, 255]
    :param x:
    :return:
    """
    return x * 127.5 + 127.5


# TODO: _sigmoid_to_255(x), _255_to_sigmoid(x)
# def _sigmoid_to_255(x):
# def _255_to_sigmoid(x):

def crop_imgs(img, ratio):
    # if Run out of CUDA memory, you can crop images to small pieces by this function. Just set --crop_scale 4
    b, c, h, w = img.shape
    h_psize = h // ratio
    w_psize = w // ratio
    imgs = torch.zeros(ratio ** 2, c, h_psize, w_psize)

    for i in range(ratio):
        for j in range(ratio):
            imgs[i * ratio + j] = img[0, :, i * h_psize:(i + 1) * h_psize, j * w_psize:(j + 1) * w_psize]
    return imgs


def binning_imgs(img, ratio):
    n, b, c, h, w = img.shape
    output = torch.zeros((b, c, h * ratio, w * ratio))

    for i in range(ratio):
        for j in range(ratio):
            output[:, :, i * h:(i + 1) * h, j * w:(j + 1) * w] = img[i * ratio + j]
    return output


def rgb_avg_downsample(rgb, scale=2):
    """
    :param rgb:rgb
    :param scale:scale
    :return: down-sampled rgb
    """
    if len(rgb.shape) < 4:
        rgb = torch.from_numpy(np.transpose(rgb, [2, 0, 1]))
        c, h, w = rgb.shape
        rgb = rgb.contiguous().view(-1, 3, h, w)

    b, c, h, w = rgb.shape

    # AvgPool2d

    avgpool = nn.AvgPool2d(scale, stride=scale)

    return avgpool(rgb)


def generate_gaussian_noise(h, w, noise_level):

    return torch.randn([1, h, w]).mul_(noise_level)


def add_noise(img, noise_level):

    noise = torch.randn(img.size()).mul_(noise_level)
    return img + noise


def rgb_downsample(img, downsampler, scale):
    """
    :param img: (batch, 3, h, w)
    :param downsampler: bicubic, avg
    :return: noisy raw
    """
    noise = np.random.normal(scale=noise_level, size=img.shape)
    noise = torch.from_numpy(noise).float()
    return img + noise


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
