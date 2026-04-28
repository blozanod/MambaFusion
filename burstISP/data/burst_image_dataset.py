from torch.utils import data as data
from burstISP.utils import img2tensor
from burstISP.utils.registry import DATASET_REGISTRY

import os
import torch
import numpy as np
import glob
import random
import pickle as pkl
import cv2

@DATASET_REGISTRY.register()
class BurstImageDataset(data.Dataset):
    def __init__(self, opt):
        super(BurstImageDataset, self).__init__()
        self.opt = opt
        self.data_root = opt['dataroot']
        self.count = opt.get('num_frames', 14)
        
        self.burst_folders = sorted(glob.glob(os.path.join(self.data_root, '*')))
        self.lq_img_count = 14

    def __getitem__(self, index):
        burst_dir = self.burst_folders[index]
        count = self.count
        assert count <= self.lq_img_count, "Burst size must be <= images in folder"

        # Get GT image
        gt_img, pkl_file = self._get_gt_image(burst_dir)

        # Get lq frames
        lq_frames =[]
        indices = self._generate_lq_indices(count) # Fixed typo in function name
        for idx in indices:
            lq_img = self._get_lq_image(burst_dir, idx)
            lq_frames.append(lq_img)
        
        # Random Flips
        """
        if self.opt['phase'] == 'train':
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            if hflip:
                # Correct PyTorch Spatial Horizontal Flip (dim=2 is Width)
                gt_img = torch.flip(gt_img, dims=[2])
                lq_frames = [torch.flip(lq, dims=[2]) for lq in lq_frames]
            
            if vflip:
                # Correct PyTorch Spatial Vertical Flip (dim=1 is Height)
                gt_img = torch.flip(gt_img, dims=[1])
                lq_frames = [torch.flip(lq, dims=[1]) for lq in lq_frames]
        """

        # Transform to tensors (dim=0 is standard over axis=0)
        img_lqs = torch.stack(lq_frames, dim=0) # [N, C, H, W]

        return {'lq': img_lqs, 'gt': gt_img, 'lq_path': burst_dir, 'meta': pkl_file}
    
    def _get_gt_image(self, burst_dir):
        pkl_file = glob.glob(os.path.join(burst_dir, '*.pkl'))[0]
        gt_img_file = glob.glob(os.path.join(burst_dir, '*_x4_rgb.png'))[0]

        img = cv2.imread(gt_img_file, cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        image = torch.from_numpy(img.astype(np.float32)).permute(2,0,1) #[3, H, W]

        with open(pkl_file, "rb") as f:
            meta_data = pkl.load(f)

        if not meta_data.get('black_level_subtracted', False):
            image = image - 512.0
        
        image = image / 16383.0

        return image, pkl_file
    
    def _get_lq_image(self, burst_dir, img_idx):
        pkl_file = glob.glob(os.path.join(burst_dir, '*.pkl'))[0]
        lq_img_paths = sorted(glob.glob(os.path.join(burst_dir, '*_x1_*.png')))
        img_path = lq_img_paths[img_idx]

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [H/2, W/2, 4]
        image = torch.from_numpy(img.astype(np.float32)).permute(2,0,1) # [4, H/2, W/2]

        with open(pkl_file, "rb") as f:
            meta_data = pkl.load(f)

        if not meta_data.get('black_level_subtracted', False):
            image = image - 512.0
        
        image = image / 16383.0

        return image
    
    def _generate_lq_indices(self, count):
        ref_idx = 0
        if count == 1:
            return [ref_idx]

        # Generate other indices
        indices = random.sample(range(1, self.lq_img_count), count - 1)
        
        indices.insert(count // 2, ref_idx)

        return indices

    def __len__(self):
        return len(self.burst_folders)