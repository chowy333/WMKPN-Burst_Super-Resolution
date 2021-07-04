import torch.nn.functional as F
import cv2
from datasets.synthetic_burst_train_set import SyntheticBurst
from torch.utils.data.dataloader import DataLoader
from utils.metrics import PSNR
from utils.postprocessing_functions import SimplePostProcess
from utils.data_format_utils import convert_dict
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from TorchTools.ArgsTools.test_args import TestArgs

def main():

    parser = argparse.ArgumentParser(description='PyTorch implementation of BurstSR')
    parsers = TestArgs()
    args = parsers.initialize(parser)
    if args.show_info:
        parsers.print_args()

    ################## Set gpu ID ########################
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    # load model architecture
    print('===> Loading the network ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(device)

    if args.show_info:
        print(model)
        print_model_parm_nums(model)



    zurich_raw2rgb = ZurichRAW2RGB(root=args.test_path, split='test')
    dataset = SyntheticBurst(zurich_raw2rgb, burst_size=args.num_seq, crop_sz=args.patch_size)

    data_loader = DataLoader(dataset, batch_size=1) # 원래 2로 되어있었다.

    # Function to calculate PSNR. Note that the boundary pixels (40 pixels) will be ignored during PSNR computation
    psnr_fn = PSNR(boundary_ignore=40)

    # Postprocessing function to obtain sRGB images
    postprocess_fn = SimplePostProcess(return_np=True)

    #model.eval()
    with torch.no_grad():
        for d in data_loader:
            burst, frame_gt, flow_vectors, meta_info = d

            # A simple baseline which upsamples the base image using bilinear   upsampling
            #burst_rgb = burst[:, 0, [0, 1, 3]]
            #burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
            #burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')

            #ours model
            bst = burst.permute(0,2,1,3,4).to(device)
            target = frame_gt.to(device)
            burst_rgb = model(bst)
            #Calculate PSNR
            score = psnr_fn(burst_rgb, frame_gt)

            print('PSNR is {:0.3f}'.format(score))

            meta_info = convert_dict(meta_info, burst.shape[0])

            # Apply simple post-processing to obtain RGB images
            pred_0 = postprocess_fn.process(burst_rgb[0], meta_info[0])
            gt_0 = postprocess_fn.process(frame_gt[0], meta_info[0])

            pred_0 = cv2.cvtColor(pred_0, cv2.COLOR_RGB2BGR)
            gt_0 = cv2.cvtColor(gt_0, cv2.COLOR_RGB2BGR)

            # Visualize input, ground truth
            #cv2.imshow('Input (Demosaicekd + Upsampled)', pred_0)
            #cv2.imshow('GT', gt_0)

            #input_key = cv2.waitKey(0)
            #if input_key == ord('q'):
            #    return


if __name__ == '__main__':
    main()
