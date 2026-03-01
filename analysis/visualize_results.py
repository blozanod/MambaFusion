# --- LICENSE NOTICE FOR DERIVED CODE ---
# Portions of this code (specifically the function 'generate_processed_image') are adapted from:
# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 
# Original source: https://github.com/goutamgmb/deep-burst-sr/blob/master/dataset/burstsr_dataset.py
# --- END LICENSE NOTICE ---

import os
import cv2
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def generate_processed_image(im, meta_data, return_np=False, external_norm_factor=None, gamma=True, smoothstep=True,
                             no_white_balance=False):
    """Processes the raw tensor into a viewable RGB image."""
    im = im * meta_data.get('norm_factor', 1.0)

    if not meta_data.get('black_level_subtracted', False):
        im = (im - torch.tensor(meta_data['black_level'])[[0, 1, -1]].view(3, 1, 1))

    if not meta_data.get('while_balance_applied', False) and not no_white_balance:
        im = im * torch.tensor(meta_data['cam_wb'])[[0, 1, -1]].view(3, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]
    
    """ TODO: Implement CCM that works for all images, different xyz2srgb matrix needs to be used, as img is already rgb
              preferrably, matrix will either be universal or use matadata from the .pkl file to determine which matrix to use
              
    # Color Correction Matrix (CCM)
    if 'rgb_xyz_matrix' in meta_data:
        # Standard XYZ to linear sRGB conversion matrix (D65) using NumPy
        xyz2srgb = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ], dtype=np.float32)

        # Extract XYZ -> Camera matrix from metadata
        raw_matrix = np.array(meta_data['rgb_xyz_matrix'], dtype=np.float32)
        xyz2cam = raw_matrix[[0, 1, 3], :]

        # Invert to get Camera -> XYZ
        cam2xyz = np.linalg.inv(xyz2cam)

        # Handle White Balance interaction
        if not no_white_balance:
            # Slice [0, 1, 3] to grab R, G1, and B, making it a 3-element array
            wb = np.array(meta_data['cam_wb'], dtype=np.float32)[[0, 1, 3]] 
            gains = wb / wb[1] # The gains applied in step 2
            inv_gains = np.diag(1.0 / gains) # Now this safely creates a 3x3 matrix
        else:
            inv_gains = np.eye(3, dtype=np.float32)

        # Compute Combined Matrix
        ccm = xyz2srgb @ cam2xyz @ inv_gains

        # Convert the final 3x3 NumPy matrix back to a tensor to apply to the image
        ccm_tensor = torch.from_numpy(ccm).to(im.device)

        # Apply Matrix
        C, H, W = im.shape
        im_flat = im.view(C, -1)     # Flatten to (3, N)
        im_flat = torch.mm(ccm_tensor, im_flat) # Matrix Multiply
        im = im_flat.view(C, H, W)   # Reshape back
        """

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


def process_pipeline(folder_path, im_name, meta_name, output_name, visualize=False):
    im_path = os.path.join(folder_path, im_name)
    meta_path = os.path.join(folder_path, meta_name)
    output_path = os.path.join(folder_path, output_name)

    # Load Image
    print(f"Loading {im_path}...")
    im_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
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
    rgb_image = generate_processed_image(im_tensor, meta_data, return_np=True)

    # 5. Save Image
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_image)
    print(f"Success! Processed image saved to: {output_path}")

    # Visualize the image using Matplotlib
    if visualize:
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title("Processed Image (RGB)")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    process_pipeline(folder_path = r"C:\Users\lozan\Documents\Education\MambaFusion\dataset\022_0047_RAW", im_name="022_MFSR_Sony_0047_x4_rgb.png", meta_name="MFSR_Sony_0047_x4.pkl", output_name="im_processed_rgb.png", visualize=True)