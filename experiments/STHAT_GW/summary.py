import re
import numpy as np

def generate_statistics_markdown(log_file_path, output_md_path):
    with open(log_file_path, 'r') as f:
        log_content = f.read()

    # Match all training iterations and loss variables
    train_pattern = re.compile(r'iter:\s*([0-9,]+).*?l_pix:\s*([0-9\.e\-]+)\s*l_edge:\s*([0-9\.e\-]+)')
    train_matches = train_pattern.findall(log_content)
    
    # Group continuous iteration stats into 5000-iteration blocks
    blocks = {}
    for match in train_matches:
        iteration = int(match[0].replace(',', ''))
        l_pix = float(match[1])
        l_edge = float(match[2])
        
        block_idx = (iteration - 1) // 5000
        if block_idx not in blocks:
            blocks[block_idx] = {'l_pix': [], 'l_edge': []}
        blocks[block_idx]['l_pix'].append(l_pix)
        blocks[block_idx]['l_edge'].append(l_edge)

    # Match validation metrics directly block by block
    val_pattern = re.compile(
        r'Validation RealBSR_val.*?# psnr_srgb:\s*([0-9\.]+)\s*Best:\s*([0-9\.]+).*?'
        r'# psnr_linear:\s*([0-9\.]+)\s*Best:\s*([0-9\.]+).*?'
        r'# ssim:\s*([0-9\.]+)\s*Best:\s*([0-9\.]+)', re.DOTALL)
    val_matches = val_pattern.findall(log_content)

    # Export statistical findings to structured markdown output
    with open(output_md_path, 'w') as f:
        f.write("### Training Validation & Loss Statistics\n\n")
        f.write("| Iteration Block | Phase | Avg `l_pix` ± Std Dev | Avg `l_edge` ± Std Dev | PSNR-SRGB | PSNR-Linear | SSIM |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        for idx in sorted(blocks.keys()):
            start_iter = idx * 5000 + 1
            end_iter = (idx + 1) * 5000
            
            pix_mean = np.mean(blocks[idx]['l_pix'])
            pix_std = np.std(blocks[idx]['l_pix'])
            edge_mean = np.mean(blocks[idx]['l_edge'])
            edge_std = np.std(blocks[idx]['l_edge'])
            
            # Identify LR warmup phase dynamically 
            phase = "LR Warmup" if end_iter <= 10000 else "Training"
            
            val_srgb, val_linear, val_ssim = "-", "-", "-"
            if idx < len(val_matches):
                match = val_matches[idx]
                
                srgb, best_srgb = float(match[0]), float(match[1])
                lin, best_lin = float(match[2]), float(match[3])
                ssim, best_ssim = float(match[4]), float(match[5])
                
                # Apply bold formatting if a recorded metric establishes a new best 
                val_srgb = f"**{srgb:.4f}**" if srgb >= best_srgb else f"{srgb:.4f}"
                val_linear = f"**{lin:.4f}**" if lin >= best_lin else f"{lin:.4f}"
                val_ssim = f"**{ssim:.4f}**" if ssim >= best_ssim else f"{ssim:.4f}"

            f.write(f"| {start_iter:,} - {end_iter:,} | {phase} | {pix_mean:.6f} ± {pix_std:.6f} | {edge_mean:.6f} ± {edge_std:.6f} | {val_srgb} | {val_linear} | {val_ssim} |\n")

if __name__ == "__main__":
    # Substitute log path mapping natively
    generate_statistics_markdown('train_STHAT_GW_20260608_214821.log', 'training_statistics.md')