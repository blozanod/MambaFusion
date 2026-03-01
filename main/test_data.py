# test_data.py, verify that burst_image_dataset is working correctly by loading a few samples and printing their shapes
import sys
import os
import matplotlib.pyplot as plt

# Ensure the root and parent directories are in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(current_dir)
sys.path.append(parent_dir)

from burstISP.data.burst_image_dataset import BurstImageDataset

if __name__ == '__main__':
    # Set up path from main/test_data.py to MambaFusion/dataset folder
    data_path = os.path.join(parent_dir, 'dataset')

    # Dummy options dictionary
    opt = {
        'dataroot': data_path, 
        'count': 5
    }

    print("Initializing dataset...")
    try:
        dataset = BurstImageDataset(opt)
        print(dataset.data_root)
        print(f"Found {len(dataset)} burst folders.")
        
        if len(dataset) > 0:
            data_item = dataset[0]
            print("\nSuccess! Output shapes:")
            print(f"LQ Tensor Shape (N, C, H, W): {data_item['lq'].shape}")
            print(f"GT Tensor Shape (C, H, W):    {data_item['gt'].shape}")
        else:
            print("No folders found. Check your 'dataroot' path.")
            
    except Exception as e:
        print(f"Error occurred during testing: {e}")

    
    # Visualize the first frame of the first burst to verify packed input
    first_frame = data_item['lq'][0] 

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Bayer Pattern Channels (RGGB)', fontsize=16)

    # Channel 0: Red
    axs[0, 0].imshow(first_frame[0], cmap='Reds')
    axs[0, 0].set_title('Channel 0 (Red)')
    axs[0, 0].axis('off')

    # Channel 1: Green 1
    axs[0, 1].imshow(first_frame[1], cmap='Greens')
    axs[0, 1].set_title('Channel 1 (Green 1)')
    axs[0, 1].axis('off')

    # Channel 2: Green 2
    axs[1, 0].imshow(first_frame[2], cmap='Greens')
    axs[1, 0].set_title('Channel 2 (Green 2)')
    axs[1, 0].axis('off')

    # Channel 3: Blue
    axs[1, 1].imshow(first_frame[3], cmap='Blues')
    axs[1, 1].set_title('Channel 3 (Blue)')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()