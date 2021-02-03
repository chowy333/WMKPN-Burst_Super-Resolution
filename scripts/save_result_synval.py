import torch.nn.functional as F
import cv2
import os
from datasets.synthetic_burst_val_set import SyntheticBurstVal
import torch
import numpy as np
import torch.nn as nn
from TorchTools.ArgsTools.test_args import TestArgs
from model.common import ConvBlock_2D, ConvBlock_3D, Ref_pad, RFAB, RTAB
import argparse
import importlib

class SimpleBaseline:
    def __init__(self):
        pass

    def __call__(self, burst):
        burst_rgb = burst[:, 0, [0, 1, 3]]
        burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
        burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')
        return burst_rgb

def main():

    parser = argparse.ArgumentParser(description='test of BurstSR')
    parsers = TestArgs()
    args = parsers.initialize(parser)
    if args.show_info:
        parsers.print_args()

    ################## Set gpu ID ########################
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    # TODO Set your network here
    #net = SimpleBaseline()

    # load model architecture
    print('===> Loading the network ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(device)

    if args.show_info:
        print(model)
        print_model_parm_nums(model)


    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=====> loading checkpoint '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            best_psnr = checkpoint['best_psnr']
            model.load_state_dict(checkpoint['state_dict'])
            print("The pretrained_model is at checkpoint {}, and it's best loss is {}."
                  .format(checkpoint['iter'], best_psnr))
        else:
            print("=====> no checkpoint found at '{}'".format(args.pretrained_model))



    val_dir = "./syn_burst_val/"
    dataset = SyntheticBurstVal(val_dir)
    out_dir = './val_out_last_conv/'
    #device = 'cuda'
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(len(dataset)):
        burst, burst_name = dataset[idx]


        #burst = burst.to(device).unsqueeze(0)
        burst = burst.unsqueeze(0).permute(0,2,1,3,4).to(device)
        #print(burst.shape)
        with torch.no_grad():
            #net_pred = net(burst)
            net_pred = model(burst)
        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)

        # Save predictions as png
        cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), net_pred_np)


if __name__ == '__main__':
    main()
