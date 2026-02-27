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


def main():
    # 1. Point this to the folder containing your im_raw.png and meta_info.pkl
    # Based on your previous error, this seems to be the right path:
    folder_path = r"C:\Users\lozan\Documents\Education\MambaFusion\dataset\000_0023_RAW"
    
    im_path = os.path.join(folder_path, '000_MFSR_Sony_0023_x4_rgb.png')
    meta_path = os.path.join(folder_path, 'MFSR_Sony_0023_x4.pkl')
    output_path = os.path.join(folder_path, 'im_processed_rgb.png')

    # 2. Load the image exactly how the CanonImage class did it
    print(f"Loading {im_path}...")
    im_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im_raw is None:
        print(f"Error: Could not find image at {im_path}")
        return
        
    im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
    im_tensor = torch.from_numpy(im_raw).float()

    # 3. Load the metadata
    print(f"Loading {meta_path}...")
    with open(meta_path, "rb") as f:
        meta_data = pkl.load(f)

    # 4. Process the image
    print("Processing image...")
    rgb_image = generate_processed_image(im_tensor, meta_data, return_np=True)

    # 5. Save the image (OpenCV expects BGR instead of RGB, so we convert before saving)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_image)
    print(f"Success! Processed image saved to: {output_path}")

    # 6. Pop up a window to visualize it immediately using Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title("Processed Canon Image (RGB)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()