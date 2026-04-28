import os
import sys
import glob
import argparse
import yaml
import copy
import torch
import cv2
import numpy as np
import pickle as pkl

# Add the parent directory to sys.path to access the burstISP modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from burstISP.archs.mambafusion_arch import MambaFusionNet
from burstISP.utils.img_util import img2tensor, imwrite, imfrombytes

# Import the ISP pipeline for visualization
# Note: Ensure this path matches exactly where you saved generate_processed_image_channel3
from burstISP.utils.img_util import generate_processed_image_channel3 

def main():
    parser = argparse.ArgumentParser(description='Run inference and visualize results using ISP pipeline.')
    parser.add_argument('--config', type=str, default='../experiments/MambaFusion_x4/config.yml', 
                        help='Path to the experiment config file to load network parameters.')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the .pth model weights file.')
    parser.add_argument('--input_dir', type=str, default='../dataset/RealBSR_RAW_testpatch/', 
                        help='Directory containing the burst images.')
    parser.add_argument('--output_path', type=str, default='./inferences_vis/', 
                        help='Path to save the visually processed images.')
    args = parser.parse_args()

    # 1. Load configuration
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
    
    network_opt = opt.get('network_g', {})
    num_frames = network_opt.get('num_frames', 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Initialize the MambaFusion architecture
    print(f"Initializing MambaFusionNet on {device}...")
    model = MambaFusionNet(**network_opt)
    
    # 3. Load model weights
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema'] # Prefer EMA weights if available!
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 4. Target directories
    input_dirs =["010_0023", "010_0104", "013_0265", "010_0292", "020_0543",
                  "014_0674", "006_0291", "007_0065", "020_0047", "027_0388"]
                  
    os.makedirs(args.output_path, exist_ok=True)

    for directory in input_dirs:
        input_path = os.path.join(args.input_dir, directory)
        print(f"\n---> Processing burst: {directory}")
        lq_img_paths = sorted(glob.glob(os.path.join(input_path, '*_x1_*.png')))
    
        if not lq_img_paths:
            lq_img_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))

        if len(lq_img_paths) < num_frames:
            print(f"Skipping {directory}: Not enough frames.")
            continue
        
        # Load metadata
        pkl_file = glob.glob(os.path.join(input_path, '*.pkl'))[0]
        with open(pkl_file, "rb") as f:
            meta_data = pkl.load(f)

        # Frame selection logic
        true_center_idx = len(lq_img_paths) // 2
        other_indices = list(range(len(lq_img_paths)))
        other_indices.remove(true_center_idx)

        np.random.seed(42)
        indices = np.random.choice(other_indices, min(num_frames - 1, len(other_indices)), replace=False)
        indices = sorted(indices)

        ref_idx = num_frames // 2
        indices.insert(ref_idx, true_center_idx)

        # --- 5. Data Loading: STRICT MATCH TO TRAINING DATALOADER ---
        lq_frames = []
        for idx in indices:
            lq_path = lq_img_paths[idx]
            with open(lq_path, 'rb') as f:
                img_lq = imfrombytes(f.read(), float32=False, flag='unchanged')
                
            img_lq = img_lq.astype(np.float32)

            # Match training: subtract 512 directly
            if not meta_data.get('black_level_subtracted', False):
                img_lq = img_lq - 512.0
            
            # Match training: normalize by 16383.0 (14-bit)
            img_lq = img_lq / 16383.0
            
            lq_frames.append(img_lq)
            
        tensor_imgs = img2tensor(lq_frames, bgr2rgb=False, float32=True)
        input_tensor = torch.stack(tensor_imgs, dim=0).unsqueeze(0).to(device)
        
        # --- 6. Run Inference ---
        print(f"Running inference. Input shape: {input_tensor.shape}")
        with torch.no_grad():
            # Support both mixed precision and standard precision models
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_tensor = model(input_tensor)
            
        # Extract tensor if the model returns a tuple
        if isinstance(output_tensor, tuple):
            output_tensor = output_tensor[0]
            
        output_tensor = output_tensor.squeeze(0).float() # [3, H, W]

        # --- 7. ISP Post-Processing ---
        print("Applying ISP pipeline to linear RGB output...")
        try:
            # We pass black_level_substracted=True because the model learned to 
            # output data with the black level ALREADY subtracted (since input had it subtracted).
            # The ISP function will apply White Balance, Exposure, Gamma, and Smoothstep.
            vis_img = generate_processed_image_channel3(
                output_tensor, 
                copy.deepcopy(meta_data), 
                return_np=True, 
                black_level_substracted=True 
            )
        except Exception as e:
            print(f"Error during ISP processing: {e}")
            continue
        
        # --- 8. Save output ---
        save_filename = os.path.join(args.output_path, f"{directory}_vis.png")
        
        # Convert RGB to BGR for cv2 saving
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        imwrite(vis_img_bgr, save_filename)
        
        print(f"Output visually saved to: {os.path.abspath(save_filename)}")

if __name__ == '__main__':
    main()