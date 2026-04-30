import torch
import glob
import os

# Get all model files and sort them numerically by iteration
files = sorted(glob.glob("net_g_*.pth"), key=lambda x: int(x.split('_')[-1].split('.')[0]))

print(f"{'Checkpoint':<20} | {'Alpha Value':<12}")
print("-" * 35)

for ckpt_path in files:
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Handle different potential keys for the state_dict
        state_dict = checkpoint.get('params', checkpoint)
        
        # Pull alpha_residual (checking for module prefix from DistributedDataParallel)
        alpha_val = state_dict.get('module.alpha_residual', state_dict.get('alpha_residual'))
        
        if alpha_val is not None:
            print(f"{ckpt_path:<20} | {alpha_val.item():.6f}")
        else:
            print(f"{ckpt_path:<20} | Not found")
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
