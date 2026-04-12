# --- LICENSE NOTICE FOR DERIVED CODE ---
# Portions of this code (specifically the function 'generate_processed_image') are adapted from:
# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 
# Original source: https://github.com/goutamgmb/deep-burst-sr/blob/master/dataset/burstsr_dataset.py
# --- END LICENSE NOTICE ---

import os
from pathlib import Path
import cv2
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# FIX 1: Removed the unused 'folder_path' argument from this function signature
def generate_processed_image(im, meta_data, return_np=False, external_norm_factor=None, gamma=True, smoothstep=True,
                             no_white_balance=False):
    """Processes the raw tensor into a viewable RGB image."""
    im = im * meta_data.get('norm_factor', 1.0)

    if not meta_data.get('black_level_subtracted', False):
        im = (im - torch.tensor(meta_data['black_level'])[[0, 1, -1]].view(3, 1, 1))

    if not meta_data.get('while_balance_applied', False) and not no_white_balance:
        im = im * torch.tensor(meta_data['cam_wb'])[[0, 1, -1]].view(3, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]
    
    """ TODO: Implement CCM that works for all images... (Commented out in original) """

    im_out = im

    if external_norm_factor is None:
        im_out = im_out / (im_out.mean() * 5.0)
    else:
        im_out = im_out / external_norm_factor

    im_out = im_out.clamp(0.0, 1.0)

    if gamma:
        im_out = im_out ** (1.0 / 2.2)

    if smoothstep:
        # Smooth curve
        im_out = 3 * im_out ** 2 - 2 * im_out ** 3

    if return_np:
        im_out = im_out.permute(1, 2, 0).numpy() * 255.0
        im_out = im_out.astype(np.uint8)
    return im_out

# FIX 2: Updated parameters to accept exact paths directly instead of piecing them together here
def process_pipeline(im_path, meta_path, output_path, visualize=False):
    # Load Image
    print(f"Loading {im_path}...")
    im_raw = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)
    if im_raw is None:
        print(f"Error: Could not find image at {im_path}")
        return
        
    im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
    im_tensor = torch.from_numpy(im_raw).float()

    # Load Metadata
    print(f"Loading {meta_path}...")
    with open(meta_path, "rb") as f:
        meta_data = pkl.load(f)

    # Process Image
    print("Processing image...")
    # FIX 3: Removed 'folder_path' from this call to match the updated signature
    rgb_image = generate_processed_image(im_tensor, meta_data, return_np=True)

    # Save Image
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)
    print(f"Success! Processed image saved to: {output_path}")

    # Visualize the image using Matplotlib
    if visualize:
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title("Processed Image (RGB)")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    input_folder = Path("inferences/V4_Current")
    
    # FIX 4: Created a dedicated output folder so original files are not overwritten
    output_folder = Path("inferences/V4_Current")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Iterate over all PNG files in the folder
    for img_path in input_folder.glob("*.png"):
        im_name = img_path.name  # e.g., '006_0291_restored.png'

        # Extract the base image name and format output
        base_name = im_name.replace('_restored.png', '')
        output_name = f"{base_name}_restored.png"
        output_path = output_folder / output_name

        # Extract the image ID (e.g., '0291' from '006_0291')
        img_id = base_name.split('_')[-1]
        
        # Construct the exact relative path to the .pkl file
        meta_dir = Path(f"../dataset/Inference_Set/{base_name}")
        meta_filename = f"MFSR_Sony_{img_id}_x4.pkl"
        meta_path = meta_dir / meta_filename

        # Call pipeline with explicit paths
        process_pipeline(
            im_path=img_path,
            meta_path=meta_path,
            output_path=output_path,
            visualize=False
        )