import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.LogTools.logger_tensorboard import Tf_Logger
from TorchTools.LossTools.loss import C_Loss
from TorchTools.LossTools.metrics import PSNR, AverageMeter


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
    criterion = nn.torch.nn.MSELoss().cuda(args.gpu_id)
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

    # When input size is constant, True is good. But opposite case is bad.
    cudnn.benchmark = True

    ############### Data loading code incompleted!!! ###################
    traindir = os.path.join(args.data, 'train') #How to load data
    valdir = os.path.join(args.data, 'val') #How to load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    current_iter = 0
    cur_psnr = 0
    cur_ssim = 0
    best_psnr = 0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    logger = Tf_Logger(args.logdir)
    losses = AverageMeter('Loss', ':.4e')
    print('---------- Start training -------------')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            current_iter += 1

            # evaluate on validation set
            if current_iter % args.valid_freq == 0:
                cur_psnr, cur_ssim = validate(val_loader, model, criterion, args)
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target).cuda()

            # measure accuracy and record loss

            losses.update(loss.item(), images.size(0))

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
                          'Loss {loss.val: .4f} ({loss.avg: .4f})\t'.format(
                           epoch, i, len(train_loader), current_iter, loss=losses))

            # log
            info = {
                'loss': loss.item(),
                'psnr': cur_psnr,
                'ssim': cur_ssim
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, current_iter)

            if current_iter % args.save_freq == 0:
                    is_best = (cur_psnr > best_psnr)
                    best_psnr = max(cur_psnr, best_psnr)
                    #model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'iter': current_iter,
                        'arch': args.model,
                        'state_dict': model.state_dict(),
                        'best_psnr': best_psnr,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, args = args)

        print('Saving the final model.')




def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    psnrs = AverageMeter("PSNR", ':.4e')
    ssims = AverageMeter("SSIM", ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            #PSNR & SSIM
            mse = criterion(output, target).item()
            psnr = PSNR(mse)
            # Add SSIM Function
            psnrs.update(psnr, gt.size(0))
            ssims.update(ssim, gt.size(0))
            print('Iter: [{0}][{1}/{2}]//[{3}/{4}]\t''TEST PSNR//Test SSIM: ({psnrs.val: .4f}, {psnrs.avg: .4f}) \t''TEST PSNR: ({ssims.val: .4f}, {ssims.avg: .4f} \t'.format(
                   iter, i, len(valid_loader), psnrs=psnrs, ssims=ssims))

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return psnrs.avg, ssims.avg

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
