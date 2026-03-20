"""
stitch_results.py
-----------------
Walks the RealBSR_RAW_testpatch dataset, applies the linear→sRGB pipeline
to every x4_rgb patch, and stitches all patches for each whole-image ID
into a single output image.

Folder convention:
  <root>/
    <patch>_<image>_RAW/
        <patch>_MFSR_Sony_<image>_x4_rgb.png   ← SR patch to process
        MFSR_Sony_<image>_x4.pkl                ← metadata

  - <image> is a 4-digit string that identifies the whole image (e.g. "0047")
  - <patch> is a 3-digit string that identifies the patch position  (e.g. "022")

Patch layout:
  Patches are numbered row-by-row, starting at 001.
  The number of columns is inferred from the set of patches that exist
  (or estimated from the max patch index via a square-root heuristic when
  patches are missing).  A blank (black) tile is inserted for any gap.

Usage:
  python stitch_results.py --dataset /path/to/RealBSR_RAW_testpatch \
                           --output  /path/to/output_dir
  
  Optional flags:
    --cols N          Force N columns instead of auto-detecting
    --no-gamma        Skip gamma correction
    --no-smoothstep   Skip smoothstep tone curve
    --ext png         Output file extension (png or jpg)
"""

import os
import re
import math
import argparse
import pickle as pkl
from collections import defaultdict

import cv2
import torch
import numpy as np
from visualize_results import generate_processed_image


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

FOLDER_RE = re.compile(r'^(\d{3})_(\d{4})(?:_RAW)?$')


def discover_patches(dataset_root):
    """
    Returns a dict:
      { image_id (str) : { patch_id (int) : folder_path (str) } }
    """
    images = defaultdict(dict)
    for entry in sorted(os.scandir(dataset_root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        m = FOLDER_RE.match(entry.name)
        if not m:
            continue
        patch_id = int(m.group(1))
        image_id = m.group(2)
        images[image_id][patch_id] = entry.path
    return images


def find_patch_file(folder, patch_id, image_id):
    """
    Looks for  <patch>_MFSR_Sony_<image>_x4_rgb.png  inside folder.
    Returns the path if found, else None.
    """
    candidate = os.path.join(
        folder,
        f"{patch_id:03d}_MFSR_Sony_{image_id}_x4_rgb.png"
    )
    if os.path.isfile(candidate):
        return candidate

    # Fallback: scan directory for any *_x4_rgb.png
    for fname in os.listdir(folder):
        if fname.endswith('_x4_rgb.png'):
            return os.path.join(folder, fname)

    return None


def find_pkl_file(folder, image_id):
    """
    Looks for  MFSR_Sony_<image>_x4.pkl  inside folder.
    Returns the path if found, else None.
    """
    candidate = os.path.join(folder, f"MFSR_Sony_{image_id}_x4.pkl")
    if os.path.isfile(candidate):
        return candidate

    # Fallback: any .pkl
    for fname in os.listdir(folder):
        if fname.endswith('.pkl'):
            return os.path.join(folder, fname)

    return None


# ---------------------------------------------------------------------------
# Grid inference
# ---------------------------------------------------------------------------

def infer_grid(patch_ids, forced_cols=None):
    """
    Given a collection of (1-based) patch IDs, return (n_rows, n_cols).

    Strategy:
      1. If forced_cols is given, use it.
      2. Otherwise try to find the smallest n_cols such that
         floor(max_id / n_cols) × n_cols covers all patches cleanly.
      3. Fall back to ceil(sqrt(max_id)).
    """
    max_id = max(patch_ids)

    if forced_cols is not None:
        n_cols = forced_cols
        n_rows = math.ceil(max_id / n_cols)
        return n_rows, n_cols

    # Heuristic: try column counts from 1..max_id
    best = None
    for c in range(1, max_id + 1):
        r = math.ceil(max_id / c)
        if best is None or abs(r - c) < abs(best[0] - best[1]):
            best = (r, c)
        # Prefer roughly square grids; stop when we overshoot
        if c > r:
            break

    return best


# ---------------------------------------------------------------------------
# Single-patch processing
# ---------------------------------------------------------------------------

def process_patch(im_path, pkl_path):
    """
    Loads and processes one patch.
    Returns an HxWx3 uint8 numpy array (RGB), or None on error.
    """
    # Load image
    im_raw = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im_raw is None:
        print(f"  [WARN] Could not read image: {im_path}")
        return None

    # BGR → RGB channel order, then to float tensor (C, H, W)
    #im_rgb = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB) if im_raw.ndim == 3 else im_raw
    im_np = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
    im_tensor = torch.from_numpy(im_np).float()

    # Load metadata
    with open(pkl_path, 'rb') as f:
        meta_data = pkl.load(f)

    # Process
    rgb = generate_processed_image(im_tensor, meta_data,
                                   return_np=True)
    return rgb  # HxWx3 uint8


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def stitch_image(patch_map, n_rows, n_cols, patch_h, patch_w):
    """
    Builds the full stitched image.
    patch_map: { patch_id (int) : folder_path }
    Missing patches are filled with black.
    Returns an HxWx3 uint8 numpy array (RGB).
    """
    canvas = np.zeros((n_rows * patch_h, n_cols * patch_w, 3), dtype=np.uint8)

    # Determine image_id from the folder name of any patch
    sample_folder = next(iter(patch_map.values()))
    m = FOLDER_RE.match(os.path.basename(sample_folder))
    image_id = m.group(2) if m else "0000"

    for row in range(n_rows):
        for col in range(n_cols):
            patch_id = col * n_rows + row + 1  # 1-based, column-major order

            y0, y1 = row * patch_h, (row + 1) * patch_h
            x0, x1 = col * patch_w, (col + 1) * patch_w

            if patch_id not in patch_map:
                # Missing patch → black tile (already zero)
                print(f"  [INFO] Patch {patch_id:03d} missing — leaving black.")
                continue

            folder = patch_map[patch_id]
            im_path = find_patch_file(folder, patch_id, image_id)
            pkl_path = find_pkl_file(folder, image_id)

            if im_path is None:
                print(f"  [WARN] No x4_rgb image found in {folder}")
                continue
            if pkl_path is None:
                print(f"  [WARN] No .pkl metadata found in {folder}")
                continue

            rgb = process_patch(im_path, pkl_path)
            if rgb is None:
                continue

            ph, pw = rgb.shape[:2]
            if ph != patch_h or pw != patch_w:
                print(f"  [WARN] Patch {patch_id} size {pw}x{ph} != expected "
                      f"{patch_w}x{patch_h}. Resizing.")
                rgb = cv2.resize(rgb, (patch_w, patch_h))

            canvas[y0:y1, x0:x1] = rgb

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process and stitch RealBSR_RAW_testpatch patches into full images."
    )
    parser.add_argument('--dataset', required=True,
                        help="Path to the RealBSR_RAW_testpatch directory.")
    parser.add_argument('--output', required=True,
                        help="Directory where stitched images will be saved.")
    parser.add_argument('--cols', type=int, default=None,
                        help="Force a fixed number of columns (default: auto-detect).")
    parser.add_argument('--no-gamma', action='store_true',
                        help="Disable gamma correction.")
    parser.add_argument('--no-smoothstep', action='store_true',
                        help="Disable smoothstep tone curve.")
    parser.add_argument('--ext', default='png', choices=['png', 'jpg'],
                        help="Output file extension (default: png).")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Scanning dataset: {args.dataset}")
    images = discover_patches(args.dataset)

    if not images:
        print("No patch folders found. Check the --dataset path and folder naming.")
        return

    print(f"Found {len(images)} unique image(s): {sorted(images.keys())}\n")

    for image_id, patch_map in sorted(images.items()):
        print(f"=== Image {image_id} — {len(patch_map)} patch(es) found ===")

        patch_ids = sorted(patch_map.keys())
        n_rows, n_cols = infer_grid(patch_ids, forced_cols=args.cols)
        print(f"  Grid: {n_rows} rows × {n_cols} cols  "
              f"(patches 1–{n_rows * n_cols}, {n_rows * n_cols - len(patch_ids)} missing)")

        # Determine patch dimensions from the first readable patch
        patch_h = patch_w = None
        sample_image_id = image_id  # used for filename lookup
        for pid in patch_ids:
            folder = patch_map[pid]
            m = FOLDER_RE.match(os.path.basename(folder))
            if m:
                sample_image_id = m.group(2)
            ip = find_patch_file(folder, pid, sample_image_id)
            if ip:
                probe = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
                if probe is not None:
                    patch_h, patch_w = probe.shape[:2]
                    print(f"  Patch size: {patch_w}×{patch_h} px")
                    break

        if patch_h is None:
            print(f"  [ERROR] Could not read any patch for image {image_id}. Skipping.\n")
            continue

        stitched = stitch_image(patch_map, n_rows, n_cols,
                                patch_h, patch_w)

        out_name = f"stitched_{image_id}.{args.ext}"
        out_path = os.path.join(args.output, out_name)

        # Convert RGB → BGR for OpenCV saving
        bgr = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr)
        print(f"  Saved → {out_path}  ({stitched.shape[1]}×{stitched.shape[0]} px)\n")

    print("Done.")


if __name__ == '__main__':
    main()