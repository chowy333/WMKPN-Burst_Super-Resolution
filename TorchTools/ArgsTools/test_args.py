import os
import numpy as np
import shutil

class TestArgs():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--gpu_id', default='2', type=str, help='GPU Index')
        parser.add_argument('--show_info', action='store_true', help='Show arg information')
        # model args
        parser.add_argument('--test_path', type=str,default='./data')
        parser.add_argument('--save_output_path', type=str,default='', help="save validation image path")
        parser.add_argument('--model', type=str, default='ours')
        parser.add_argument('--denoise', action='store_true',  help='denoise store_true')
        parser.add_argument('--norm_type', default=None, type=str,
                            help='dm_block_type(default: rrdb)')
        parser.add_argument('--block_type', default='rrdb', type=str,
                            help='dm_block_type(default: rrdb)')
        parser.add_argument('--act_type', default='relu', type=str,
                            help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--bias', action='store_true',
                            help='bias of layer')
        parser.add_argument('--filters', default=64, type=int,
                            help='filters')
        parser.add_argument('--reduction', default=8, type=int,
                            help='Squeeze and Excitation')
        parser.add_argument('--in_channels', default=4, type=int, help='in_channels')
        parser.add_argument('--gt_channels', default=3, type=int, help='gt_channels')
        parser.add_argument('--pretrained_model', default='', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--n_resblocks', default=6, type=int,
                            help='number of basic blocks')
        parser.add_argument('--num_rfab', default=8, type=int,
                            help='number of  residual feature attention block')
        parser.add_argument('--sr_n_resblocks', default=6, type=int,
                            help='number of demosaicking blocks')
        parser.add_argument('--dm_n_resblocks', default=6, type=int,
                            help='number of demosaicking blocks')
        # datasets args
        parser.add_argument('--num_seq', default=14, type=int, help='Number of Burst Sequence (default: 14)')
        parser.add_argument('--patch_size', default=384, type=int, help='height of patch (default: 384)')

        # for super-resolution
        parser.add_argument('--scale', default=4, type=int,
                            help='Scale of Super-resolution.')
        parser.add_argument('--downsampler', default='avg', type=str,
                            help='downsampler of Super-resolution.')

        self.args = parser.parse_args()

        return self.args

    def print_args(self):
        # print args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("\n")
        print("==========     CONFIG END    =============")
