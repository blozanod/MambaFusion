import os
import sys
import argparse
import yaml
import copy
import torch
import cv2
import random
import numpy as np
import pickle as pkl
import glob
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from burstISP.archs.mambafusion_arch import MambaFusionNet
from burstISP.utils.img_util import img2tensor, imwrite, imfrombytes
from burstISP.utils.img_util import generate_processed_image_channel3 
from burstISP.data.burst_image_dataset import BurstImageDataset

def get_step_number(filename):
    """Extracts the numerical step from the checkpoint filename."""
    match = re.search(r'\d+', os.path.basename(filename))
    return match.group() if match else "unknown_step"

def main():
    parser = argparse.ArgumentParser(description='Run batch inference across multiple checkpoints for visual progress tracking.')
    parser.add_argument('--config', type=str, default='../experiments/MambaFusion_x4/config.yml')
    parser.add_argument('--checkpoints_dir', type=str, required=True, help='Directory containing the model .pth files')
    parser.add_argument('--input_dir', type=str, default='../dataset/RealBSR_RAW_testpatch/')
    parser.add_argument('--output_path', type=str, default='./progress_vis/')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
    
    network_opt = opt.get('network_g', {})
    num_frames = network_opt.get('num_frames', 5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Gather and sort checkpoints sequentially
    ckpt_paths = glob.glob(os.path.join(args.checkpoints_dir, '*.pth'))
    # Sort numerically based on the step extracted from the filename
    ckpt_paths.sort(key=lambda x: int(get_step_number(x)) if get_step_number(x).isdigit() else 0)

    if not ckpt_paths:
        print(f"Error: No .pth files found in {args.checkpoints_dir}")
        return

    print(f"Found {len(ckpt_paths)} checkpoints. Initializing MambaFusionNet on {device}...")
    model = MambaFusionNet(**network_opt)
    model.to(device)

    input_dirs = ["010_0023", "010_0104", "013_0265", "010_0292", "020_0543",
                  "014_0674", "006_0291", "007_0065", "020_0047", "027_0388"]
                  
    os.makedirs(args.output_path, exist_ok=True)

    dataset_opt = {'dataroot': args.input_dir, 'num_frames': num_frames, 'phase': 'test'}
    dataset = BurstImageDataset(dataset_opt)
    random.seed(42)

    # 2. Optimization: Pre-find the dataset indices for the target images 
    # so we don't iterate the entire dataset for every checkpoint.
    print("Scanning dataset for target images to optimize I/O...")
    target_indices = []
    for i in range(len(dataset)):
        # We load just enough to check the path
        data = dataset[i]
        directory = os.path.basename(data['lq_path'])
        if directory in input_dirs:
            target_indices.append((i, directory))
            if len(target_indices) == len(input_dirs):
                break # Found all of them

    # 3. Outer loop: Iterate over checkpoints
    for ckpt_path in ckpt_paths:
        step = get_step_number(ckpt_path)
        print(f"\n{'='*50}")
        print(f"Loading checkpoint: {os.path.basename(ckpt_path)} (Step: {step})")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint.get('state_dict', checkpoint)))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # 4. Inner loop: Iterate over the specific target images
        for idx, directory in target_indices:
            print(f"  -> Processing burst: {directory}")
            
            # Create specific image subfolder
            img_out_folder = os.path.join(args.output_path, directory)
            os.makedirs(img_out_folder, exist_ok=True)

            data = dataset[idx]
            with open(data['meta'], "rb") as f:
                meta_data = pkl.load(f)

            input_tensor = data['lq'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output_tensor = model(input_tensor)
                
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            output_tensor = output_tensor.squeeze(0).float() 

            try:
                vis_img = generate_processed_image_channel3(
                    output_tensor, 
                    copy.deepcopy(meta_data), 
                    return_np=True, 
                    black_level_substracted=True 
                )
            except Exception as e:
                print(f"  -> Error during ISP processing for {directory}: {e}")
                continue
            
            # Title the image as the checkpoint step
            save_filename = os.path.join(img_out_folder, f"{step}.png")
            
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            imwrite(vis_img_bgr, save_filename)
            
    print(f"\nAll inferences complete. Results organized in: {os.path.abspath(args.output_path)}")

if __name__ == '__main__':
    main()