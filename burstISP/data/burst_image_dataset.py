from torch.utils import data as data

from burstISP.utils import imfrombytes, img2tensor
from burstISP.utils.registry import DATASET_REGISTRY

import os
import numpy as np
import glob

@DATASET_REGISTRY.register()
class BurstImageDataset(data.Dataset):
    """Burst image dataset for image restoration.

    Read burst LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot (str): Data root path for gt.
            count (int): Number of images to include per burst (e.g. 14 for full burst, 7 for half burst, etc)
    """

    def __init__(self, opt):
        super(BurstImageDataset, self).__init__()
        self.opt = opt
        self.data_root = opt['dataroot']
        self.count = opt['count'] if 'count' in opt else 14
        # Gets all burst folders xxx_xxxx_RAW
        self.burst_folders = sorted(glob.glob(os.path.join(self.data_root, '*_RAW')))

    def __getitem__(self, index):
        burst_dir = self.burst_folders[index]
        count = self.count
        
        # Get GT image
        gt_img_path = glob.glob(os.path.join(burst_dir, '*_x4_rgb.png'))[0]
        with open(gt_img_path, 'rb') as f:
            img_gt = imfrombytes(f.read(), float32=False, flag='unchanged')
        
        img_gt = img_gt.astype(np.float32) / 65535.0 # Normalize 16-bit image

        # Get lq image paths
        lq_frames = []
        lq_img_paths = glob.glob(os.path.join(burst_dir, '*_x1_*.png'))

        # Get indices from count random frames (sorted)
        indices = np.random.choice(len(lq_img_paths), count, replace=False)
        indices = sorted(indices)
        
        # Get lq frames
        for idx in indices:
            lq_path = lq_img_paths[idx]
            with open(lq_path, 'rb') as f:
                img_lq = imfrombytes(f.read(), float32=False, flag='unchanged')
            img_lq = img_lq.astype(np.float32) / 65535.0 # Normalize 16-bit image
            lq_frames.append(img_lq)

        # Transform to tensors
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lqs = img2tensor(lq_frames, bgr2rgb=True, float32=True)

        img_lqs = np.stack(img_lqs, axis=0) # shape: (N, C, H, W) where N is number of frames in burst

        return {'lq': img_lqs, 'gt': img_gt, 'meta': {'burst_dir': burst_dir}}
    
    def __len__(self):
        return len(self.burst_folders)