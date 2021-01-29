import argparse
import os
import random
import shutil
import time
import warnings
import importlib
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from model.common import print_model_parm_nums
from torch.utils.data import DataLoader
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.LogTools.logger_tensorboard import Tf_Logger
from TorchTools.LossTools.loss import C_Loss
from TorchTools.LossTools.metrics import AverageMeter
from utils.metrics import PSNR
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswith("__")
#    and callable(models.__dict__[name]))

def main():

    ################## parser ########################
    parser = argparse.ArgumentParser(description='PyTorch implementation of BurstSR')
    parsers = BaseArgs()
    args = parsers.initialize(parser)
    parsers.print_args()

    ################## Set gpu ID ########################
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    ################## for reproducibility ########################
    if args.seed is not None:
        os.environ['PYTHONHASHSEED']=str(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        # When input size is constant, True is good. But opposite case is bad.
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ################## Loading the network ########################
    print('===> Loading the network ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(device)
    print(model)
    print_model_parm_nums(model)

    ########## define loss function (criterion) and optimizer #############
    #criterion = nn.torch.nn.MSELoss().cuda(args.gpu_id)
    criterion = nn.L1Loss().cuda(args.gpu_id)
    #ssim_loss_func = pytorch_ssim.SSIM()
    # if args.vgg_lambda != 0:
    #     vggloss = VGGLoss(vgg_path=args.vgg_path, layers=args.vgg_layer, loss=args.vgg_loss)
    # else:
    #     vggloss = None
    #criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_adjust_freq, gamma=0.1)

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)

    ################## Loading checkpoints ########################
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


    ############### Data loading code incompleted!!! ###################
    base_data = ZurichRAW2RGB(root =args.train_path, split = "train")
    burst_data= SyntheticBurst(base_dataset=base_data) # a, b, c, d = burst_data[0] #len(burst_data) = 46839
    tr_burst, val_burst = torch.utils.data.random_split(burst_data,
                        [int(len(burst_data) * args.split_ratio), len(burst_data) - int(len(burst_data) * args.split_ratio)])

    train_loader = DataLoader(dataset=tr_burst, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_burst, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True)


    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    current_iter = 0
    cur_psnr = 0
    #cur_ssim = 0
    best_psnr = 0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    logger = Tf_Logger(args.logdir)
    losses = AverageMeter('Loss', ':.4e')

    print('---------- Start training -------------')
    for epoch in range(args.start_epoch, args.end_epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)

        #progress = ProgressMeter(
        #    len(train_loader),
        #    [batch_time, data_time, losses],
        #    prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        for i, (burst, frame_gt, flow_vectors, meta_info) in enumerate(train_loader):
            #print(len(train_loader))
            #print("frame =", frame_gt.shape)
            # burst = torch.Size([B, N, 4, 48, 48])
            # frame_gt = torch.Size([B, 3, 384, 384])
            # flow_vectors = torch.Size([B, N, 2, 96, 96])
            # torch.Size dict
            current_iter += 1

            # evaluate on validation set
            if current_iter % args.valid_freq == 0:
                #cur_psnr, cur_ssim = validate(val_loader, model, criterion, args)
                cur_psnr, val_loss_avg = validate(val_loader, model, criterion, current_iter, device, logger, args)

                val_info = {
                    'Loss/val_loss': val_loss_avg,
                }
                for tag, value in val_info.items():
                    logger.scalar_summary(tag, value, current_iter)

            # measure data loading time
            data_time.update(time.time() - end)

            model.train()
            bst = burst.permute(0, 2, 1, 3, 4).to(device)
            target = frame_gt.to(device)

            # compute output
            output = model(bst)
            #print("output", output.shape)
            loss = criterion(output, target).cuda()

            # compute the loss
            losses.update(loss.item(), burst.size(0))

            # compute gradient and do step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if current_iter % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                          'Iter:{3}\t'
                          'Loss:{loss.val:.4f} (avg:{loss.avg:.4f})\t'.format(
                           epoch, i, len(train_loader), current_iter, loss=losses))

            # log
            info = {
                'Loss/train_loss': loss.item(),
                'Accuracy/psnr': cur_psnr
                #'ssim': cur_ssim
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, current_iter)

            if current_iter % args.save_freq == 0:
                    is_best = (cur_psnr > best_psnr)
                    best_psnr = max(cur_psnr, best_psnr)
                    #model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                    save_checkpoint({
                        'epoch': epoch,
                        'iter': current_iter,
                        'arch': args.model,
                        'state_dict': model.state_dict(),
                        'best_psnr': best_psnr,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, args = args)

        print('Saving the final model.')




def validate(val_loader, model, criterion, current_iter, device, logger, args):
    losses = AverageMeter('Val_Loss', ':.4e')
    psnrs = AverageMeter("PSNR", ':.4e')
    metric = PSNR()
    #ssims = AverageMeter("SSIM", ':.4e')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (burst, frame_gt, flow_vectors, meta_info) in enumerate(val_loader):

            bst = burst.permute(0,2,1,3,4).to(device)
            target = frame_gt.to(device)

            # compute output
            output = model(bst)
            loss = criterion(output, target).cuda()

            #PSNR & SSIM
            psnr = metric(output, target)

            # Add Real PSNR, SSIM Function

            psnrs.update(psnr.item(), target.size(0))
            #ssims.update(ssim, target.size(0))

            #print('Iter: [{0}][{1}/{2}]\t''TEST PSNR:({psnrs.val: .4f}, avg:{psnrs.avg:     #.4f})\t'.format(current_iter, i, len(val_loader), psnrs=psnrs))
            #print('Iter: [{0}][{1}/{2}]\t''TEST PSNR: ({psnrs.val: .4f}, {psnrs.avg: .4f})\t''TEST #SSIM: ({ssims.val: .4f}, {ssims.avg: .4f})\t'.format(
            #        iter, i, len(val_loader), psnrs=psnrs, ssims=ssims))

            losses.update(loss.item(), burst.size(0))

            print('Iter: [{0}][{1}/{2}]\t''TEST PSNR: ({psnrs.val: .4f}, {psnrs.avg: .4f})\t''val_loss: ({losses.val: .4f}, {losses.avg: .4f})\t'.format(
                     current_iter, i, len(val_loader), psnrs=psnrs, losses=losses))
            #if i % args.print_freq == 0:
            #    progress.display(i)


        #print('*Final_validation Score* PSNR_avg {psnrs.avg:.3f}'.format(psnrs=psnrs))

        #print('*Final_validation Score* PSNR_avg {psnrs.avg:.3f} SSIM #{SSIM.avg:.3f}'.format(psnrs=psnrs, ssims=ssims))

        # show images
        output = torch.clamp(output[:4,:].cpu(), 0, 1).detach().numpy()
        target = torch.clamp(target[:4,:].cpu(), 0, 1).detach().numpy()
        #target = np.transpose(target, [1, 2, 0]) 	# permute

        vis_info = {
            'Visualize/output': output,
            'Visualize/target': target
        }

        for tag, value in vis_info.items():
            logger.image_summary(tag, value, current_iter)


        #vis = [output, target]
        #logger.image_summary("Visualize", vis, current_iter)

    #return psnrs.avg, ssims.avg
    return psnrs.avg, losses.avg

def save_checkpoint(state, is_best, args):
    filename = '{}/{}_checkpoint_{}k.path'.format(args.save_path, args.post, state['iter']/1000)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_model_best.path'.format(args.save_path, args.post))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
