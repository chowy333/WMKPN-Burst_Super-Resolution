import os
import datetime
class BaseArgs():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # datasets args
        parser.add_argument('--train_path', type=str,
                            default='/home/wooyeong/Burst/burstsr_dataset/Zurich_Public',  help='path to train')
        #parser.add_argument('--valid_path', type=str,
        #                    default='datasets/valid_df2k.txt', help='path to valid')
        parser.add_argument('--split_ratio', default=0.8, type=float, help='train dataset split ratio')

        parser.add_argument('--evaluate', action='store_true', help='test')


        parser.add_argument('--denoise', action='store_true',  help='denoise store_true')
        parser.add_argument('--max_noise', default=0.0748, type=float, help='noise_level')
        parser.add_argument('--min_noise', default=0.00, type=float, help='noise_level')

        parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default:8)')
        parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers (default: 8)')
        parser.add_argument('--patch_size', default=256, type=int, help='height of patch (default: 64)')
        parser.add_argument('--num_seq', default=14, type=int, help='height of patch (default: 64)')
        parser.add_argument('--in_channels', default=4, type=int, help='in_channels')
        parser.add_argument('--gt_channels', default=3, type=int, help='gt_channels')
        #parser.add_argument('--get2label', action='store_true',  help='denoise store_true')
        #parser.add_argument('--bayer_only', action='store_true',  help='get only raw images')


        # train args
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--end_epochs', default=10000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--total_iters', default=10000000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--start_iters', default=0, type=int,
                            help='number of total epochs to run')

        parser.add_argument('--lr_adjust_freq', default=200000, type=int,
                            help='decent the lr in epoch number')
        parser.add_argument('--lr', default=1e-4, type=float,
                            help='initial learning rate')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        # valid args
        parser.add_argument('--valid_freq', default=5000, type=int,
                            help='epoch interval for valid (default:5000)')
        # logger parse
        parser.add_argument('--print_freq', default=100, type=int,
                            help='print frequency (default: 100)')
        parser.add_argument('--postname', type=str, default='',
                            help='postname of save model')
        parser.add_argument('--save_path', type=str, default='./ckpts',
                            help='path of save model')
        parser.add_argument('--save_freq', default=100000, type=int,
                            help='save ckpoints frequency (default: 100000)')
        parser.add_argument('--logdir', default='./logs', type=str,
                            help='log dir')
        # model args
        parser.add_argument('--pretrained_model', default='', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--model', default='ours', type=str,
                            help='model name (default: none)')
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

        # for single task
        parser.add_argument('--n_resblocks', default=6, type=int,
                            help='number of basic blocks')
        # for joint task
        parser.add_argument('--num_rfab', default=8, type=int,
                            help='number of  residual feature attention block')
        parser.add_argument('--sr_n_resblocks', default=6, type=int,
                            help='number of demosaicking blocks')
        parser.add_argument('--dm_n_resblocks', default=6, type=int,
                            help='number of demosaicking blocks')
        # for super-resolution
        parser.add_argument('--scale', default=4, type=int,
                            help='Scale of Super-resolution.')
        parser.add_argument('--downsampler', default='avg', type=str,
                            help='downsampler of Super-resolution.')
        # loss args
        parser.add_argument('--vgg_path', default='/mnt/lustre/qianguocheng/codefiles/vgg19.pth',
                            type=str, help='vgg loss pth location')
        parser.add_argument('--vgg_loss', type=str, default='l1', help="loss L1 or L2 ['l1', 'l2']")
        parser.add_argument('--vgg_layer', type=str, default='5', help='number of threads to prepare data.')
        parser.add_argument('--dm_lambda', type=float, default=0.5, help='dm loss lamda')
        parser.add_argument('--sr_lambda', type=float, default=1, help='sr loss lamda')
        parser.add_argument('--ssim_lambda', type=float, default=0, help='ssim loss lamda')
        parser.add_argument('--vgg_lambda', type=float, default=0, help='vgg loss lamda, set 0 if no vgg_loss')

        parser.add_argument('--remove', action='store_true', help='remove save_path')
        parser.add_argument('--flag', type=str, default=None)
        parser.add_argument('--bayer_mode', type=str, default='rggb', help="['rggb', 'grbg']")
        parser.add_argument('--seed', default=0, type=int, help='seed')
        parser.add_argument('--gpu_id', default='2', type=str, help='GPU Index')
        parser.add_argument('--cl_weights', default='1,1,1,1', type=str, help='curriculum learning weight')

        parser.add_argument('--num_steps', default=4, type=int, help='feedback count')
        parser.add_argument('--custom_lr_path', type=str, default=None, help='path of custom lr image')
        parser.add_argument('--custom_lr_train_dir', type=str, default='PixelShift200_train_crop_lr')
        parser.add_argument('--custom_lr_valid_dir', type=str, default='PixelShift200_valid_crop_lr')


        # for test jupyter
        self.args = parser.parse_args()
        # for test jupyter
        # self.args = parser.parse_args(args[0)

        if self.args.denoise:
            self.args.post = self.args.model + '-dn-'+ self.args.train_path.split('/')[-1]\
                             +'-'+str(self.args.num_rfab)\
                             +'-'+str(self.args.filters) + '-' +'x'+ str(self.args.scale)
        else:
            self.args.post = self.args.model + self.args.train_path.split('/')[-1]\
                             +'-'+str(self.args.num_rfab)\
                             +'-'+str(self.args.filters) + '-' +'x'+ str(self.args.scale)
        if self.args.postname:
            self.args.post = self.args.post +'-' + self.args.postname
        if self.args.flag:
            self.args.post = self.args.post +'-' + self.args.flag

        self.initialized = True
        return self.args

    def print_args(self):
        # making foot print
        now = datetime.datetime.now()
        fw = open("footprint_{}.txt".format(self.args.post), "w")
        fw.write("Datetime : {}\n".format(now))
        fw.close()
        # print args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("\n")
        print("==========     CONFIG END    =============")




        # check for folders existence
        if os.path.exists(os.path.join(self.args.logdir, self.args.post)):
            cmd = 'rm -rf ' + os.path.join(self.args.logdir, self.args.post)
            os.system(cmd)
        os.makedirs(os.path.join(self.args.logdir, self.args.post))

        if not os.path.exists(os.path.join(self.args.save_path, self.args.post)):
            os.makedirs(os.path.join(self.args.save_path, self.args.post))

        assert os.path.exists(self.args.train_path), 'train_list {} not found'.format(self.args.train_path)
        #assert os.path.exists(self.args.valid_path), 'valid_list {} not #found'.format(self.args.valid_path)
