import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Ensure the root and parent directories are in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(current_dir)
sys.path.append(parent_dir)

from burstISP.data.burst_image_dataset import BurstImageDataset
from burstISP.archs.mambafusion_arch import MambaFusionNet

def test_full_cycle(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Full Cycle Test on {device} ---")

    # 1. Initialize Dataset and Dataloader
    try:
        dataset = BurstImageDataset(opt)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        batch = next(iter(dataloader))
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)
        print(f"Data loaded. LQ: {lq.shape}, GT: {gt.shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 2. Initialize Model
    # Note: opt['is_train'] is used in the model's __init__
    model = MambaFusionNet(**opt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    # --- PART A: TRAINING PASS ---
    print("\n[Step 1/2] Testing Training Pass...")
    model.train()
    # Ensure the internal flag is set to True
    model.is_train = True 
    
    try:
        optimizer.zero_grad()
        
        # In train mode, we expect 3 outputs
        output, aligned_burst, fusion_output = model(lq)
        
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        
        print("✓ Training Forward Pass Successful (3 outputs received)")
        print("✓ Backward Pass and Optimizer Step Successful")
        print(f"  Loss: {loss.item():.6f}")
    except Exception as e:
        print(f"❌ Training Pass Failed: {e}")

    # --- PART B: INFERENCE/TESTING PASS ---
    print("\n[Step 2/2] Testing Inference Pass...")
    model.eval()
    # Explicitly toggle the flag to False for the return logic
    model.is_train = False 
    
    try:
        with torch.no_grad():
            # In eval mode, we expect ONLY 1 output
            # This mimics what happens during validation
            result = model(lq)
            
            # Check if it returned a tuple anyway (which is what crashed the CRC)
            if isinstance(result, tuple):
                print(f"⚠️ Warning: Model returned a tuple of size {len(result)} even in eval mode!")
                output_image = result[0]
            else:
                print("✓ Model returned a single Tensor (correct behavior for inference)")
                output_image = result

            print(f"✓ Inference Output Shape: {output_image.shape}")
            
    except Exception as e:
        print(f"❌ Inference Pass Failed: {e}")

if __name__ == '__main__':
    data_path = os.path.join(parent_dir, 'dataset/RealBSR_RAW_testpatch_testing')

    opt = {
        'is_train': True,
        'dataroot': data_path,
        'img_size': 80,
        'num_frames': 5,
        'num_feat': 64,
        'out_chans': 3,
        'offset_groups': 4,
        'fusion_heads': 4,
        'd_state': 16,
        'window_size': 16,
        'inner_rank': 32,
        'num_tokens': 8,
        'embed_dim': 64,
        'resi_connection': '1conv',
        'convffn_kernel_size': 5,
        'scale': 8,
        'depths': [3, 3, 3, 3],
        'num_heads': [4, 4, 4, 4], # Set to 4 to be divisible by 64
        'mlp_ratio': 1,
        'upsampler': 'pixelshuffledirect'
    }

    test_full_cycle(opt)