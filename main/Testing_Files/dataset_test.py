import os
import sys
import torch
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))

sys.path.append(current_dir)
sys.path.append(parent_dir)

from burstISP.data.burst_image_dataset import BurstImageDataset
from burstISP.archs.mambafusion_arch import GlobalSkipConnection
from burstISP.utils.img_util import generate_processed_image_channel3

def run_smoke_test(dataroot):
    print("Starting Smoke Test...")
    
    # 1. Initialize the Dataset
    opt = {
        'dataroot': dataroot,
        'num_frames': 7,     # Test with half-burst (7 frames)
        'phase': 'val'       # 'val' disables the random flips
    }
    
    try:
        dataset = BurstImageDataset(opt)
        print(f"Dataset initialized! Found {len(dataset)} bursts.")
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    # 2. Get the first burst
    data = dataset[0]
    lq_burst = data['lq']  # [N, 4, H/2, W/2]
    gt_img = data['gt']    # [3, 2H, 2W]
    pkl_path = data['meta']
    
    print(f"Loaded burst data.")
    print(f"   LQ Burst Shape: {lq_burst.shape}")
    print(f"   GT Image Shape: {gt_img.shape}")
    
    # Load metadata
    with open(pkl_path, 'rb') as f:
        meta_data = pickle.load(f)

    # 3. Extract the Reference Frame
    # By dataset design, the reference frame is placed exactly at index `count // 2`
    ref_idx = opt['num_frames'] // 2
    lq_ref = lq_burst[ref_idx] #[4, H/2, W/2]
    print(f"Extracted reference frame at index {ref_idx}.")

    # DELETED: The BGR->RGB RAW Channel Contamination fix. 
    # The channels natively load as [R, G1, G2, B] correctly.
    
    # 4. Run Global Skip Connection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Add batch dimension and send directly to device
    lq_tensor = lq_ref.unsqueeze(0).to(device) #[1, 4, H/2, W/2]
    gt_tensor = gt_img.to(device)
    
    # Initialize Skip Connection (scale=4 maps [H/2, W/2] to [2H, 2W])
    skip_model = GlobalSkipConnection(scale=8).to(device)
    
    with torch.no_grad():
        lq_baseline = skip_model(lq_tensor) #[1, 3, 2H, 2W]
        lq_baseline = lq_baseline.squeeze(0) #[3, 2H, 2W]
    
    print(f"Ran Global Skip Connection. Output Shape: {lq_baseline.shape}")

    # 6. Post-Process Both Images through ISP Pipeline
    # Note: dataset already did `image - 512.0`, so we set black_level_substracted=True
    try:
        # Clone meta_data to prevent accidental modifications between calls
        import copy
        
        lq_vis = generate_processed_image_channel3(
            lq_baseline, 
            copy.deepcopy(meta_data), 
            return_np=True, 
            black_level_substracted=True
        )
        
        gt_vis = generate_processed_image_channel3(
            gt_tensor, 
            copy.deepcopy(meta_data), 
            return_np=True, 
            black_level_substracted=True
        )
        print("ISP Post-Processing completed.")
    except Exception as e:
        print(f"ISP Pipeline crashed: {e}")
        return

    # 7. Save Output to Disk
    # OpenCV expects BGR to write to disk, so we convert RGB -> BGR
    lq_save_path = "smoke_test_LQ_baseline.png"
    gt_save_path = "smoke_test_GT_target.png"
    
    cv2.imwrite(lq_save_path, cv2.cvtColor(lq_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(gt_save_path, cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))
    
    print(f"Success! Images saved to '{lq_save_path}' and '{gt_save_path}'.")
    
    # Optional: If you are in a Jupyter Notebook, you can plot them:
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0].imshow(lq_vis)
        axes[0].set_title("LQ Baseline (Global Skip Connection)")
        axes[0].axis('off')
        
        axes[1].imshow(gt_vis)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
    except Exception:
        pass # Matplotlib not configured/installed, totally fine.

if __name__ == '__main__':
    TEST_DATAROOT = os.path.join(parent_dir, 'dataset/RealBSR_RAW_testpatch_testing') 
    
    if not os.path.exists(TEST_DATAROOT):
        print(f"Please update TEST_DATAROOT. Path '{TEST_DATAROOT}' not found.")
    else:
        run_smoke_test(TEST_DATAROOT)