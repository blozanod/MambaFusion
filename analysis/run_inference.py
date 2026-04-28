import os
import sys
import glob
import argparse
import yaml
import torch
import cv2
import numpy as np
import pickle as pkl

# Add the parent directory to sys.path to access the burstISP modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from burstISP.archs.mambafusion_arch import MambaFusionNet
from burstISP.utils.img_util import img2tensor, tensor2img, imwrite, imfrombytes

def main():
    parser = argparse.ArgumentParser(description='Run inference using a trained MambaFusion model.')
    parser.add_argument('--config', type=str, default='../experiments/MambaFusion_x4/config.yml', 
                        help='Path to the experiment config file to load network parameters.')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the .pth model weights file (from the models folder).')
    parser.add_argument('--input_dir', type=str, default='../dataset/RealBSR_RAW_testpatch/', 
                        help='Directory containing the burst images for a single test patch.')
    parser.add_argument('--output_path', type=str, default='./inferences/', 
                        help='Path to save the restored image.')
    args = parser.parse_args()

    # 1. Load configuration
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
    
    network_opt = opt.get('network_g', {})
    num_frames = network_opt.get('num_frames', 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Initialize the MambaFusion architecture
    print(f"Initializing MambaFusionNet...")
    model = MambaFusionNet(**network_opt)
    
    # 3. Load model weights
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract state dict (handles formats saved directly or nested under 'params')
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 4. Load the input burst frames
    input_dirs =["010_0023","010_0104","013_0265","010_0292","020_0543",
                  "014_0674","006_0291","007_0065","020_0047","027_0388"]
    for directory in input_dirs:
        input_path = os.path.join(args.input_dir, directory)
        print(f"Loading input burst from {input_path}...")
        lq_img_paths = sorted(glob.glob(os.path.join(input_path, '*_x1_*.png')))
    
        # Fallback just in case the folder doesn't use the standard RealBSR naming convention
        if not lq_img_paths:
            lq_img_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))

        if len(lq_img_paths) < num_frames:
            raise ValueError(f"Expected at least {num_frames} images in {args.input_dir}, but found {len(lq_img_paths)}.")
        
        # Load metadata (.pkl) to match training data preparation
        pkl_file = glob.glob(os.path.join(input_path, '*.pkl'))[0]
        with open(pkl_file, "rb") as f:
            meta_data = pkl.load(f)

        bl = np.array(meta_data['black_level'], dtype=np.float32) / 65535.0
        wb = np.array(meta_data['cam_wb'], dtype=np.float32)
        wb_gain = wb / wb[1]

        # Get indices for requested frames, isolating the true center frame
        true_center_idx = len(lq_img_paths) // 2
        other_indices = list(range(len(lq_img_paths)))
        other_indices.remove(true_center_idx)

        # Fix the random seed for reproducible inferences
        np.random.seed(42)
        indices = np.random.choice(other_indices, min(num_frames - 1, len(other_indices)), replace=False)
        indices = sorted(indices)

        ref_idx = num_frames // 2
        indices.insert(ref_idx, true_center_idx)  # Insert center frame index in the middle

        lq_frames =[]
        for idx in indices:
            lq_path = lq_img_paths[idx]
            with open(lq_path, 'rb') as f:
                img_lq = imfrombytes(f.read(), float32=False, flag='unchanged')
                
            # Normalize 16-bit image to [0, 1] range as done in BurstImageDataset
            img_lq = img_lq.astype(np.float32) / 65535.0 

            # Apply metadata to input to match training perfectly
            if not meta_data.get('black_level_subtracted', False):
                img_lq = img_lq - bl.reshape(1, 1, 4)
            if not meta_data.get('while_balance_applied', False):
                img_lq = img_lq * wb_gain.reshape(1, 1, 4)
            
            img_lq = np.clip(img_lq, 0.0, 1.0)
            lq_frames.append(img_lq)
            
        # Convert images to tensors, ensuring we DO NOT scramble the channels (bgr2rgb=False)
        tensor_imgs = img2tensor(lq_frames, bgr2rgb=False, float32=True)
        
        # Stack the burst into [1, N, C, H, W] for the model forward pass
        input_tensor = torch.stack(tensor_imgs, dim=0).unsqueeze(0).to(device)
        
        # 5. Run Inference
        print(f"Running inference. Input shape: {input_tensor.shape}")
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # 6. Post-process and save output
        print("Processing output...")

        os.makedirs(args.output_path, exist_ok=True)
        save_filename = os.path.join(args.output_path, f"{directory}_restored.png")
        
        # Manually convert [0, 1] tensor back to a 16-bit numpy array
        if isinstance(output_tensor, tuple):
            output_tensor = output_tensor[0]
        
        output_img = output_tensor.squeeze(0).float().detach().cpu().clamp_(0, 1)
        output_img = (output_img.numpy() * 65535.0).round().astype(np.uint16)
        
        # Rearrange channels from (C, H, W) to (H, W, C) for cv2
        output_img = output_img.transpose(1, 2, 0)
        
        # Convert RGB to BGR for cv2 saving
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # Create directories and save the 16-bit image
        imwrite(output_img, save_filename)
        print(f"Output saved successfully to {os.path.abspath(save_filename)}")

if __name__ == '__main__':
    main()