import os
import glob

import torch
import torchvision.transforms as tfm
import imageio

class Div2k(torch.utils.data.Dataset):
    """Div2k dataset class
    args: 
        dir_data: root of dataset
    """
    def __init__(self,dir_data='/home/wooyeong/samsung/dataset/'):
        self.dir_data = dir_data
        _set_filesystem(self.dir_data)
        self.images_hr = _scan()

    def __len__(self):
        return len(self.images_hr)

    def __getitem(self, idx):
        f_hr = self.images_hr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)

        return hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.ext = '.png'

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext))
        )
        return names_hr
    
        